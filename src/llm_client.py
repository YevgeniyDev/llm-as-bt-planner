"""
LLM client wrapper for converting natural-language instructions into plans.

The planner uses in-context learning so the model sees examples of the exact
action vocabulary expected by the behavior tree builder. This reduces semantic
drift between the LLM output and the downstream executor.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_HUGGINGFACE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
PLACEHOLDER_SECRETS = {
    "your_huggingface_token_here",
    "your_openai_api_key_here",
}
ACTION_SIGNATURE_PATTERN = re.compile(r"^\s*([A-Za-z_ ]+?)\s*(?:\((.*?)\))?\s*$")


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


REACTIVE_COMPILER_NOTES = """
The downstream compiler turns your symbolic plan into a reactive behavior tree.

Compilation rules:
- Pick(object) is guarded by a holding condition.
- MoveTo(target) is guarded by an at-location condition.
- Place(object, target) is guarded by an object-at-target condition and may
  recover by ensuring the robot is holding the object and standing at target.
- Insert(object, target) is guarded by an inserted condition and may recover by
  ensuring the robot is holding the object and standing at target.

Therefore, prefer plans with clear postconditions and sensible object/target
pairs so the compiled reactive BT behaves predictably.
""".strip()


RECURSIVE_SYSTEM_PROMPT = """
You are implementing Algorithm 1 style recursive BT planning.

Given a task instruction, the current predicted robot state, and the remaining
recursion budget, decide whether to:
- return a primitive manipulation plan using only Pick, Place, MoveTo, Insert
- or decompose the task into smaller ordered subgoals

Decision rules:
1. Use "primitive" when the task can be expressed directly as a short symbolic plan.
2. Use "decompose" for multi-object, multi-location, or clearly multi-stage tasks.
3. If remaining depth is 1 or less, prefer "primitive".
4. Return only valid JSON with the required keys.
5. Keep subgoals concrete, ordered, and close to the original wording.
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


RECURSIVE_DECISION_SCHEMA = {
    "name": "recursive_bt_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["primitive", "decompose"],
            },
            "reason": {"type": "string"},
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
            },
            "subgoals": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["kind", "reason", "plan", "subgoals"],
        "additionalProperties": False,
    },
}


@dataclass
class RecursiveDecision:
    kind: str
    reason: str
    plan: List[Dict[str, str]]
    subgoals: List[str]


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

    def plan_task(
        self,
        instruction: str,
        state_summary: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Convert a free-form user instruction into a validated action list.

        Even though the model returns a wrapped JSON object for compatibility,
        this method returns only the ordered action array because that is the
        structure consumed by the behavior tree builder.
        """

        cleaned_instruction = instruction.strip()
        if not cleaned_instruction:
            raise ValueError("Instruction must be a non-empty string.")

        response = self._request_completion(
            self._build_plan_messages(cleaned_instruction, state_summary=state_summary),
            json_schema=PLAN_SCHEMA,
        )
        payload = self._parse_json_response(response)
        plan = self._canonicalize_plan(self._extract_plan(payload))
        self._validate_plan(plan)
        return plan

    def revise_plan(
        self,
        instruction: str,
        current_plan: List[Dict[str, str]],
        human_feedback: str,
        tree_preview: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Revise a previously generated plan using human-in-the-loop feedback.
        """

        cleaned_instruction = instruction.strip()
        cleaned_feedback = human_feedback.strip()
        if not cleaned_instruction:
            raise ValueError("Instruction must be a non-empty string.")
        if not cleaned_feedback:
            raise ValueError("Human feedback must be a non-empty string.")

        response = self._request_completion(
            self._build_revision_messages(
                instruction=cleaned_instruction,
                current_plan=current_plan,
                human_feedback=cleaned_feedback,
                tree_preview=tree_preview,
            ),
            json_schema=PLAN_SCHEMA,
        )
        payload = self._parse_json_response(response)
        plan = self._canonicalize_plan(self._extract_plan(payload))
        self._validate_plan(plan)
        return plan

    def choose_recursive_expansion(
        self,
        instruction: str,
        state_summary: str,
        remaining_depth: int,
        max_subgoals: int = 4,
    ) -> RecursiveDecision:
        """
        Decide whether to decompose a task further or emit a primitive plan.
        """

        cleaned_instruction = instruction.strip()
        if not cleaned_instruction:
            raise ValueError("Instruction must be a non-empty string.")

        response = self._request_completion(
            self._build_recursive_messages(
                instruction=cleaned_instruction,
                state_summary=state_summary,
                remaining_depth=remaining_depth,
                max_subgoals=max_subgoals,
            ),
            json_schema=RECURSIVE_DECISION_SCHEMA,
        )
        payload = self._parse_json_response(response)
        return self._parse_recursive_decision(payload)

    def _build_plan_messages(
        self,
        instruction: str,
        state_summary: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Create the standard planning prompt.
        """

        user_lines = ['Create a task plan for this instruction: "{}"'.format(instruction)]
        if state_summary:
            user_lines.extend(
                [
                    "Current predicted world state:",
                    state_summary,
                    "Use this predicted state to keep the plan coherent.",
                ]
            )

        return [
            {
                "role": "system",
                "content": "{}\n\n{}".format(SYSTEM_PROMPT, REACTIVE_COMPILER_NOTES),
            },
            {
                "role": "user",
                "content": "\n".join(user_lines),
            },
        ]

    def _build_recursive_messages(
        self,
        instruction: str,
        state_summary: str,
        remaining_depth: int,
        max_subgoals: int,
    ) -> List[Dict[str, str]]:
        """
        Create the recursive decomposition prompt.
        """

        return [
            {
                "role": "system",
                "content": "{}\n\n{}".format(RECURSIVE_SYSTEM_PROMPT, REACTIVE_COMPILER_NOTES),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        'Task instruction: "{}"'.format(instruction),
                        "Current predicted world state:",
                        state_summary,
                        "Remaining recursion depth: {}".format(remaining_depth),
                        "Maximum subgoals to emit: {}".format(max_subgoals),
                        "Return JSON with keys kind, reason, plan, subgoals.",
                    ]
                ),
            },
        ]

    def _build_revision_messages(
        self,
        instruction: str,
        current_plan: List[Dict[str, str]],
        human_feedback: str,
        tree_preview: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Create a plan-repair prompt for Scheme 3 human-in-the-loop revision.
        """

        user_message = [
            'The original instruction is: "{}"'.format(instruction),
            "The current JSON plan is:",
            json.dumps(current_plan, indent=2),
            "A human reviewer said the current reactive BT is not correct.",
            "Human feedback: {}".format(human_feedback),
        ]

        if tree_preview:
            user_message.extend(
                [
                    "The current reactive BT preview is:",
                    tree_preview,
                ]
            )

        user_message.append(
            "Revise the plan so the compiled reactive BT better satisfies the instruction and the human feedback."
        )

        return [
            {
                "role": "system",
                "content": "{}\n\n{}".format(SYSTEM_PROMPT, REACTIVE_COMPILER_NOTES),
            },
            {
                "role": "user",
                "content": "\n".join(user_message),
            },
        ]

    def _parse_recursive_decision(self, payload: Any) -> RecursiveDecision:
        """
        Normalize a recursive decomposition decision into a typed helper object.
        """

        if not isinstance(payload, dict):
            raise ValueError("Recursive planning response must be a JSON object.")

        kind = payload.get("kind")
        if not isinstance(kind, str) or kind not in {"primitive", "decompose"}:
            raise ValueError("Recursive planning response is missing a valid 'kind'.")

        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            reason = ""

        raw_plan = payload.get("plan", [])
        plan = self._canonicalize_plan(raw_plan) if isinstance(raw_plan, list) else []
        if plan:
            self._validate_plan(plan)

        raw_subgoals = payload.get("subgoals", [])
        subgoals = []
        if isinstance(raw_subgoals, list):
            for subgoal in raw_subgoals:
                if isinstance(subgoal, str) and subgoal.strip():
                    subgoals.append(subgoal.strip())

        return RecursiveDecision(
            kind=kind,
            reason=reason.strip(),
            plan=plan,
            subgoals=subgoals,
        )

    def _parse_json_response(self, response: Any) -> Union[Dict[str, Any], List[Any]]:
        """
        Normalize a model response into generic JSON.
        """

        message = response.choices[0].message

        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError("The model refused to generate a plan: {}".format(refusal))

        content = self._coerce_message_content(message.content)
        if not content:
            raise RuntimeError("The model response did not include any content.")

        return self._parse_json_payload(content)

    def _request_completion(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        require_json: bool = True,
    ) -> Any:
        """
        Submit a planning or revision request to the configured model provider.

        We prefer Structured Outputs for `gpt-4o-mini`-class models because the
        JSON schema adds a reliable contract between the planner and the parser.
        For older or third-party models, we fall back to prompt-only JSON output.
        """

        if self._supports_structured_outputs() and json_schema is not None:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": json_schema},
                temperature=0.0,
            )

        if self.provider == "openai" and require_json:
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

    def _canonicalize_plan(self, plan: List[Any]) -> List[Dict[str, str]]:
        """
        Normalize mildly inconsistent model outputs into the internal plan schema.
        """

        canonical_plan = []
        for index, step in enumerate(plan, start=1):
            if isinstance(step, str) and step.strip():
                step = {"action": step.strip()}
            elif not isinstance(step, dict):
                raise ValueError("Plan step {} is not a JSON object.".format(index))
            canonical_plan.append(self._canonicalize_step(step))
        return canonical_plan

    def _canonicalize_step(self, step: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract only the action fields the BT compiler understands.
        """

        raw_action_name = step.get("action")
        if not isinstance(raw_action_name, str) or not raw_action_name.strip():
            raise ValueError("Plan step is missing a valid 'action' field.")

        parsed_action_name, parsed_arguments = self._split_action_signature(raw_action_name)
        canonical_action_name = self._canonicalize_action_name(parsed_action_name)

        canonical_step: Dict[str, str] = {"action": canonical_action_name}

        if canonical_action_name == "MoveTo":
            target_value = self._first_non_empty_string(
                step.get("target"),
                step.get("destination"),
                step.get("location"),
                step.get("object"),
                step.get("item"),
                parsed_arguments[0] if len(parsed_arguments) >= 1 else None,
            )
            if target_value is not None:
                canonical_step["target"] = target_value
            return canonical_step

        object_value = self._first_non_empty_string(
            step.get("object"),
            step.get("item"),
            parsed_arguments[0] if len(parsed_arguments) >= 1 else None,
        )

        if canonical_action_name == "Pick":
            if object_value is not None:
                canonical_step["object"] = object_value
            return canonical_step

        target_value = self._first_non_empty_string(
            step.get("target"),
            step.get("destination"),
            step.get("location"),
            parsed_arguments[1] if len(parsed_arguments) >= 2 else None,
        )

        if object_value is not None:
            canonical_step["object"] = object_value
        if target_value is not None:
            canonical_step["target"] = target_value
        return canonical_step

    def _split_action_signature(self, raw_action_name: str) -> tuple[str, List[str]]:
        """
        Split action strings like `Place(gear, tray)` into a name plus arguments.
        """

        match = ACTION_SIGNATURE_PATTERN.match(raw_action_name.strip())
        if not match:
            return raw_action_name.strip(), []

        action_name = match.group(1).strip()
        raw_arguments = match.group(2)
        if not raw_arguments:
            return action_name, []

        arguments = [part.strip() for part in raw_arguments.split(",") if part.strip()]
        return action_name, arguments

    def _canonicalize_action_name(self, raw_action_name: str) -> str:
        """
        Map equivalent action spellings onto the exact BT vocabulary.
        """

        normalized_name = raw_action_name.strip().replace("_", "").replace(" ", "").lower()
        mapping = {
            "pick": "Pick",
            "place": "Place",
            "moveto": "MoveTo",
            "insert": "Insert",
        }
        if normalized_name not in mapping:
            return raw_action_name.strip()
        return mapping[normalized_name]

    def _first_non_empty_string(self, *values: Any) -> Optional[str]:
        """
        Return the first candidate value that is a non-empty string.
        """

        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

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
