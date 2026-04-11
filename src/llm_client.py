"""
LLM client wrapper for converting natural-language instructions into plans.

The planner uses in-context learning so the model sees examples of the exact
action vocabulary expected by the behavior tree builder. This reduces semantic
drift between the LLM output and the downstream executor.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_HUGGINGFACE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
PLACEHOLDER_SECRETS = {
    "your_huggingface_token_here",
    "your_openai_api_key_here",
}


SYSTEM_PROMPT = """
You are a robotics task planner that converts a user's natural-language request
into a machine-readable manipulation plan.

You may only use the following action primitives:
- Pick
- Place
- MoveTo
- Insert

Planning rules:
1. Return only valid JSON and never include markdown, prose, or explanations.
2. The JSON must be an object with a single key called "plan".
3. The value of "plan" must be an ordered array of action objects.
4. Every action object must include an "action" field.
5. Use "object" for the manipulated item when relevant.
6. Use "target" for the destination, fixture, or assembly location when relevant.
7. If the robot must relocate before placing or inserting, include an explicit MoveTo step.
8. Stay close to the user's wording, but choose concrete, compact labels.
9. If the instruction is underspecified, produce the smallest sensible plan using only the supported actions.

Example 1
Instruction: Pick the bolt and place it on the tray.
Output:
{
  "plan": [
    {"action": "Pick", "object": "bolt"},
    {"action": "MoveTo", "target": "tray"},
    {"action": "Place", "object": "bolt", "target": "tray"}
  ]
}

Example 2
Instruction: Assemble the gearbox.
Output:
{
  "plan": [
    {"action": "Pick", "object": "gear"},
    {"action": "MoveTo", "target": "chassis"},
    {"action": "Insert", "object": "gear", "target": "chassis"}
  ]
}
""".strip()


PLAN_SCHEMA = {
    "name": "robot_task_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["Pick", "Place", "MoveTo", "Insert"],
                        },
                        "object": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["plan"],
        "additionalProperties": False,
    },
}


class LLMTaskPlanner:
    """
    Small wrapper around chat-completions compatible LLM providers.

    The class deliberately separates prompt construction, API invocation, and
    output validation so students can inspect or extend each stage independently.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.provider = self._resolve_provider(provider)
        self.model = self._resolve_model(model, self.provider)
        self.api_key = self._resolve_api_key(api_key, self.provider)
        self.client = self._build_client()

    def plan_task(self, instruction: str) -> List[Dict[str, str]]:
        """
        Convert a free-form user instruction into a validated action list.

        Even though the model returns a wrapped JSON object for compatibility,
        this method returns only the ordered action array because that is the
        structure consumed by the behavior tree builder.
        """

        cleaned_instruction = instruction.strip()
        if not cleaned_instruction:
            raise ValueError("Instruction must be a non-empty string.")

        response = self._request_plan(cleaned_instruction)
        message = response.choices[0].message

        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError("The model refused to generate a plan: {}".format(refusal))

        content = self._coerce_message_content(message.content)
        if not content:
            raise RuntimeError("The model response did not include any content.")

        payload = self._parse_json_payload(content)
        plan = self._extract_plan(payload)
        self._validate_plan(plan)
        return plan

    def _request_plan(self, instruction: str) -> Any:
        """
        Submit the planning request to the configured model provider.

        We prefer Structured Outputs for `gpt-4o-mini`-class models because the
        JSON schema adds a reliable contract between the planner and the parser.
        For older or third-party models, we fall back to prompt-only JSON output.
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": 'Create a task plan for this instruction: "{}"'.format(instruction),
            },
        ]

        if self._supports_structured_outputs():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA},
                temperature=0.0,
            )

        if self.provider == "openai":
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )

    def _resolve_provider(self, provider: Optional[str]) -> str:
        """
        Choose the LLM provider from explicit input, environment, or available tokens.
        """

        candidate = provider or os.getenv("LLM_PROVIDER")
        if candidate:
            normalized = candidate.strip().lower()
            if normalized in {"hf", "huggingface"}:
                return "huggingface"
            if normalized == "openai":
                return "openai"
            raise ValueError(
                "Unsupported LLM provider '{}'. Use 'huggingface' or 'openai'.".format(
                    candidate
                )
            )

        if self._get_config_value("HF_TOKEN", "HUGGINGFACE_API_KEY"):
            return "huggingface"

        if self._get_config_value("OPENAI_API_KEY"):
            return "openai"

        return "huggingface"

    def _resolve_model(self, model: Optional[str], provider: str) -> str:
        """
        Resolve the active model using generic or provider-specific environment variables.
        """

        if model:
            return model

        generic_model = os.getenv("LLM_MODEL")
        if generic_model:
            return generic_model

        if provider == "huggingface":
            return os.getenv("HUGGINGFACE_MODEL", DEFAULT_HUGGINGFACE_MODEL)

        return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    def _resolve_api_key(self, api_key: Optional[str], provider: str) -> str:
        """
        Resolve the API key required by the chosen provider.
        """

        if api_key:
            return api_key

        if provider == "huggingface":
            resolved_key = self._get_config_value("HF_TOKEN", "HUGGINGFACE_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "HF_TOKEN is not set. Copy .env.example to .env and add a Hugging Face token."
                )
            return resolved_key

        resolved_key = self._get_config_value("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your OpenAI API key."
            )

        return resolved_key

    def _build_client(self) -> OpenAI:
        """
        Build the chat client for the selected provider.
        """

        if self.provider == "huggingface":
            return OpenAI(api_key=self.api_key, base_url=HUGGINGFACE_BASE_URL)

        return OpenAI(api_key=self.api_key)

    def _get_config_value(self, *env_var_names: str) -> Optional[str]:
        """
        Return the first non-empty, non-placeholder environment value.
        """

        for env_var_name in env_var_names:
            value = os.getenv(env_var_name)
            if not value:
                continue

            cleaned_value = value.strip()
            if not cleaned_value:
                continue

            if cleaned_value.lower() in PLACEHOLDER_SECRETS:
                continue

            return cleaned_value

        return None

    def _supports_structured_outputs(self) -> bool:
        """
        Detect whether the selected model should use JSON schema output mode.

        Structured Outputs are enabled only for OpenAI models that advertise
        schema support. Other providers still work through prompt-constrained
        JSON and the parser fallback below.
        """

        if self.provider != "openai":
            return False

        structured_prefixes = (
            "gpt-4o-mini",
            "gpt-4o",
        )
        return self.model.startswith(structured_prefixes)

    def _coerce_message_content(self, content: Any) -> str:
        """
        Normalize SDK message content into a single string.
        """

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_chunks.append(text_value)
                    continue

                item_type = getattr(item, "type", None)
                text_value = getattr(item, "text", None)
                if item_type == "text" and isinstance(text_value, str):
                    text_chunks.append(text_value)
            return "\n".join(text_chunks).strip()

        return ""

    def _parse_json_payload(self, raw_content: str) -> Union[Dict[str, Any], List[Any]]:
        """
        Parse JSON directly or recover it from common model wrappers such as fences.
        """

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            pass

        candidate = self._extract_fenced_json(raw_content) or self._find_first_json_block(
            raw_content
        )
        if candidate is None:
            raise ValueError(
                "The model response did not contain valid JSON. Raw response: {}".format(
                    raw_content
                )
            )

        return json.loads(candidate)

    def _extract_fenced_json(self, content: str) -> Optional[str]:
        """
        Handle responses like ```json { ... } ```.
        """

        fence_marker = "```"
        start = content.find(fence_marker)
        while start != -1:
            line_end = content.find("\n", start)
            if line_end == -1:
                return None

            end = content.find(fence_marker, line_end + 1)
            if end == -1:
                return None

            fenced_body = content[line_end + 1 : end].strip()
            try:
                json.loads(fenced_body)
                return fenced_body
            except json.JSONDecodeError:
                start = content.find(fence_marker, end + len(fence_marker))

        return None

    def _find_first_json_block(self, content: str) -> Optional[str]:
        """
        Recover the first balanced JSON object or array embedded in free-form text.
        """

        opening_to_closing = {"{": "}", "[": "]"}
        for start_index, character in enumerate(content):
            if character not in opening_to_closing:
                continue

            stack = [opening_to_closing[character]]
            in_string = False
            escape_next = False

            for end_index in range(start_index + 1, len(content)):
                current = content[end_index]

                if escape_next:
                    escape_next = False
                    continue

                if current == "\\":
                    escape_next = True
                    continue

                if current == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if current in opening_to_closing:
                    stack.append(opening_to_closing[current])
                    continue

                if stack and current == stack[-1]:
                    stack.pop()
                    if not stack:
                        candidate = content[start_index : end_index + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break

        return None

    def _extract_plan(self, payload: Any) -> List[Dict[str, str]]:
        """
        Normalize the returned JSON into the plan array expected downstream.
        """

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            if "plan" in payload and isinstance(payload["plan"], list):
                return payload["plan"]
            if "steps" in payload and isinstance(payload["steps"], list):
                return payload["steps"]

        raise ValueError(
            "Unexpected plan format. Expected a list or a JSON object with 'plan' or 'steps'."
        )

    def _validate_plan(self, plan: List[Dict[str, str]]) -> None:
        """
        Perform lightweight semantic validation before tree construction.

        The builder still validates required fields per action, but validating
        here surfaces LLM formatting issues earlier in the pipeline.
        """

        if not isinstance(plan, list) or not plan:
            raise ValueError("The generated plan must be a non-empty list of action steps.")

        for index, step in enumerate(plan, start=1):
            if not isinstance(step, dict):
                raise ValueError("Plan step {} is not a JSON object.".format(index))

            action_name = step.get("action")
            if not isinstance(action_name, str) or not action_name.strip():
                raise ValueError("Plan step {} is missing a valid 'action' field.".format(index))
