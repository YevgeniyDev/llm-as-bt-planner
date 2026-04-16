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
from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI

from .gridworld_domain import (
    DEFAULT_TOOL_NAME,
    SUPPORTED_OBJECT_KINDS,
    SUPPORTED_RENDER_SHAPES,
    default_render_shape,
    infer_object_kind,
    insert_targets_for_kind,
    is_insert_compatible,
)
from .multi_robot_actions import RobotProfile
from .multi_robot_planner import build_multi_robot_tree_from_json


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
- ChangeTool
- Handoff

Planning rules:
1. Return only valid JSON and never include markdown, prose, or explanations.
2. The JSON must be an object with a single key called "plan".
3. The value of "plan" must be an ordered array of action objects.
4. Every action object must include an "action" field.
5. Use "object" for the manipulated item when relevant.
6. Use "target" for the destination, fixture, or assembly location when relevant.
7. Use "tool" for ChangeTool and for actions that require a specific tool.
8. Use "recipient" for Handoff to name the receiving robot when relevant.
9. Use "location" for the handoff meeting point when relevant.
10. If the robot must relocate before placing, inserting, or handing off, include an explicit MoveTo step.
11. Stay close to the user's wording, but choose concrete, compact labels.
12. If the instruction is underspecified, produce the smallest sensible plan using only the supported actions.

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

Example 3
Instruction: Switch to the inward gripper and insert the gear into the chassis.
Output:
{
  "plan": [
    {"action": "ChangeTool", "tool": "inward_gripper"},
    {"action": "Pick", "object": "gear"},
    {"action": "MoveTo", "target": "chassis"},
    {"action": "Insert", "object": "gear", "target": "chassis", "tool": "inward_gripper"}
  ]
}

Example 4
Instruction: Have one robot pass the gear to robot2 at the handoff station.
Output:
{
  "plan": [
    {"action": "Pick", "object": "gear"},
    {"action": "MoveTo", "target": "handoff_station"},
    {"action": "Handoff", "object": "gear", "recipient": "robot2", "location": "handoff_station"}
  ]
}
""".strip()


REACTIVE_COMPILER_NOTES = """
The downstream compiler turns your symbolic plan into a reactive behavior tree.

Compilation rules:
- Pick(object) is guarded by a holding condition.
- MoveTo(target) is guarded by an at-location condition.
- ChangeTool(tool) is guarded by an equipped-tool condition.
- Place(object, target) is guarded by an object-at-target condition and may
  recover by ensuring the robot is holding the object and standing at target.
- Insert(object, target) is guarded by an inserted condition and may recover by
  ensuring the robot is holding the object, standing at target, and equipping
  the requested tool when one is specified.
- Handoff(object, recipient, location) is guarded by the recipient already
  holding the object and may recover by ensuring the giver holds the object and
  both robots meet at the handoff location.

Therefore, prefer plans with clear postconditions and sensible object/target
pairs so the compiled reactive BT behaves predictably.
""".strip()


RECURSIVE_SYSTEM_PROMPT = """
You are implementing Algorithm 1 style recursive BT planning.

Given a task instruction, the current predicted robot state, and the remaining
recursion budget, decide whether to:
- return a primitive manipulation plan using only Pick, Place, MoveTo, Insert,
  ChangeTool, Handoff
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
                            "enum": [
                                "Pick",
                                "Place",
                                "MoveTo",
                                "Insert",
                                "ChangeTool",
                                "Handoff",
                            ],
                        },
                        "object": {"type": "string"},
                        "target": {"type": "string"},
                        "tool": {"type": "string"},
                        "recipient": {"type": "string"},
                        "location": {"type": "string"},
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
                            "enum": [
                                "Pick",
                                "Place",
                                "MoveTo",
                                "Insert",
                                "ChangeTool",
                                "Handoff",
                            ],
                        },
                        "object": {"type": "string"},
                        "target": {"type": "string"},
                        "tool": {"type": "string"},
                        "recipient": {"type": "string"},
                        "location": {"type": "string"},
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


GRIDWORLD_SYSTEM_PROMPT = """
You are a multi-robot gridworld task planner.

Convert a natural-language gridworld scenario into a structured JSON execution
spec for a local MRBTP-style simulator.

You may only use the following action primitives inside the plan:
- Pick
- Place
- MoveTo
- Insert
- ChangeTool
- Handoff

Gridworld planning rules:
1. Return only valid JSON and never include markdown, prose, or explanations.
2. Use exactly the provided robot names and include every requested robot exactly once.
3. Every plan step must include a "robot" field and an "action" field.
4. If a robot should wait, do not invent a Wait action. Give that robot no
   MoveTo steps and place it at an appropriate start_location.
5. For "different corners", distribute objects across distinct corner targets.
6. Use explicit Handoff steps whenever one robot transfers an object to another.
7. Keep the plan small and concrete: usually 2-4 movable task objects unless
   the user or interface gives an explicit count.
8. Each object entry may include:
   - "kind" for semantics such as circle, gear, shaft, pin, plate
   - "shape" for rendering: circle, square, triangle
   Fixed fixtures and stations such as chassis, bearing_block, panel_slot,
   tool_rack, and precision_station are locations, not movable objects.
9. success_conditions must describe the final intended placed/inserted state.
10. Each robot can carry at most one object at a time. Before the same robot
    receives or picks another object, it must Place, Insert, or Handoff the
    current one away.
11. After a Handoff, the giver no longer holds the object. Later steps must use
    the new holder if that object moves again.
12. Do not mix giver and receiver steps inside the same terminal segment. For a
    handoff, only the giver's Pick/MoveTo/Handoff steps should appear before the
    Handoff action. The receiver's follow-up actions come after the Handoff.
13. If the receiver needs to travel to the meeting point, you may omit that
    travel step; the MRBTP runtime handles receiver preparation.
14. If an object already reached its final robot, use MoveTo + Place instead of
    repeating a redundant Handoff from a robot that no longer owns the object.
15. A stationary robot with can_move=false must only act at its start_location.
16. If a stationary receiver accepts multiple objects at one location, serialize
    those handoffs and place each object before the next handoff starts.

Example 1
Instruction: All robots collect circles and place them in different corners.
Output:
{
  "task_summary": "Three robots distribute circles to different corners.",
  "robots": [
    {"name": "robot_1", "role": "collector", "start_location": "left_mid", "can_move": true},
    {"name": "robot_2", "role": "collector", "start_location": "upper_mid", "can_move": true},
    {"name": "robot_3", "role": "collector", "start_location": "lower_mid", "can_move": true}
  ],
  "objects": [
    {"name": "circle_1", "kind": "circle", "shape": "circle"},
    {"name": "circle_2", "kind": "circle", "shape": "circle"},
    {"name": "circle_3", "kind": "circle", "shape": "circle"}
  ],
  "plan": [
    {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
    {"robot": "robot_1", "action": "MoveTo", "target": "top_left"},
    {"robot": "robot_1", "action": "Place", "object": "circle_1", "target": "top_left"},
    {"robot": "robot_2", "action": "Pick", "object": "circle_2"},
    {"robot": "robot_2", "action": "MoveTo", "target": "top_right"},
    {"robot": "robot_2", "action": "Place", "object": "circle_2", "target": "top_right"},
    {"robot": "robot_3", "action": "Pick", "object": "circle_3"},
    {"robot": "robot_3", "action": "MoveTo", "target": "bottom_right"},
    {"robot": "robot_3", "action": "Place", "object": "circle_3", "target": "bottom_right"}
  ],
  "success_conditions": [
    {"object": "circle_1", "target": "top_left"},
    {"object": "circle_2", "target": "top_right"},
    {"object": "circle_3", "target": "bottom_right"}
  ]
}

Example 2
Instruction: Two robots collect circles and give them to the third one which just waits.
Output:
{
  "task_summary": "Two collectors hand circles to a stationary receiver at the center.",
  "robots": [
    {"name": "robot_1", "role": "collector", "start_location": "left_mid", "can_move": true},
    {"name": "robot_2", "role": "collector", "start_location": "lower_mid", "can_move": true},
    {"name": "robot_3", "role": "waiting_receiver", "start_location": "center", "can_move": false}
  ],
  "objects": [
    {"name": "circle_1", "kind": "circle", "shape": "circle"},
    {"name": "circle_2", "kind": "circle", "shape": "circle"}
  ],
  "plan": [
    {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
    {"robot": "robot_1", "action": "MoveTo", "target": "center"},
    {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_3", "location": "center"},
    {"robot": "robot_3", "action": "Place", "object": "circle_1", "target": "center"},
    {"robot": "robot_2", "action": "Pick", "object": "circle_2"},
    {"robot": "robot_2", "action": "MoveTo", "target": "center"},
    {"robot": "robot_2", "action": "Handoff", "object": "circle_2", "recipient": "robot_3", "location": "center"},
    {"robot": "robot_3", "action": "Place", "object": "circle_2", "target": "center"}
  ],
  "success_conditions": [
    {"object": "circle_1", "target": "center"},
    {"object": "circle_2", "target": "center"}
  ]
}

Example 3
Instruction: First robot hands a circle to robot 2 at the center, then robot 2 carries it to the top-left corner and gives it to robot 3 there.
Output:
{
  "task_summary": "A chained handoff moves one circle across three robots.",
  "robots": [
    {"name": "robot_1", "role": "collector", "start_location": "left_mid", "can_move": true},
    {"name": "robot_2", "role": "relay", "start_location": "center", "can_move": true},
    {"name": "robot_3", "role": "receiver", "start_location": "upper_mid", "can_move": true}
  ],
  "objects": [
    {"name": "circle_1", "kind": "circle", "shape": "circle"}
  ],
  "plan": [
    {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
    {"robot": "robot_1", "action": "MoveTo", "target": "center"},
    {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_2", "location": "center"},
    {"robot": "robot_2", "action": "MoveTo", "target": "top_left"},
    {"robot": "robot_2", "action": "Handoff", "object": "circle_1", "recipient": "robot_3", "location": "top_left"},
    {"robot": "robot_3", "action": "MoveTo", "target": "bottom_left"},
    {"robot": "robot_3", "action": "Place", "object": "circle_1", "target": "bottom_left"}
  ],
  "success_conditions": [
    {"object": "circle_1", "target": "bottom_left"}
  ]
}

Example 4
Instruction: Robot 1 switches to the precision gripper, picks the gear, and inserts it into the chassis.
Output:
{
  "task_summary": "A precision-tool insert places one gear into the chassis.",
  "robots": [
    {"name": "robot_1", "role": "assembler", "start_location": "left_mid", "can_move": true}
  ],
  "objects": [
    {"name": "gear_1", "kind": "gear", "shape": "circle"}
  ],
  "plan": [
    {"robot": "robot_1", "action": "MoveTo", "target": "tool_rack"},
    {"robot": "robot_1", "action": "ChangeTool", "tool": "precision_gripper"},
    {"robot": "robot_1", "action": "Pick", "object": "gear_1"},
    {"robot": "robot_1", "action": "MoveTo", "target": "chassis"},
    {"robot": "robot_1", "action": "Insert", "object": "gear_1", "target": "chassis", "tool": "precision_gripper"}
  ],
  "success_conditions": [
    {"object": "gear_1", "target": "chassis"}
  ]
}
""".strip()


GRIDWORLD_REPAIR_PROMPT = """
You are revising a multi-robot gridworld scenario JSON after validation failed.

Return only corrected JSON matching the same schema as before.

Fix the scenario so it is compatible with the MRBTP-style planner and the local
gridworld executor. In particular:
- one explicit giver owns each Handoff segment
- the giver must still hold the object when handing off
- do not repeat a Handoff from a robot that already gave the object away
- avoid mixing giver and receiver steps before the same Handoff action
- each robot may hold only one object at a time
- keep object kinds, render shapes, and insert targets semantically compatible
- if the same stationary receiver gets multiple objects, insert a Place step
  before the next Handoff so it is empty-handed again
- if a robot is stationary, keep all of its actions at its start_location
- keep exactly the requested robot count and exact robot names
- preserve the user's intent as closely as possible
""".strip()


GRIDWORLD_PLAN_SCHEMA = {
    "name": "gridworld_multi_robot_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "task_summary": {"type": "string"},
            "robots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "start_location": {"type": "string"},
                        "can_move": {"type": "boolean"},
                    },
                    "required": ["name", "role", "start_location", "can_move"],
                    "additionalProperties": False,
                },
            },
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": sorted(SUPPORTED_OBJECT_KINDS),
                        },
                        "shape": {
                            "type": "string",
                            "enum": sorted(SUPPORTED_RENDER_SHAPES),
                        },
                    },
                    "required": ["name", "shape"],
                    "additionalProperties": False,
                },
            },
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "robot": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": [
                                "Pick",
                                "Place",
                                "MoveTo",
                                "Insert",
                                "ChangeTool",
                                "Handoff",
                            ],
                        },
                        "object": {"type": "string"},
                        "target": {"type": "string"},
                        "tool": {"type": "string"},
                        "recipient": {"type": "string"},
                        "location": {"type": "string"},
                    },
                    "required": ["robot", "action"],
                    "additionalProperties": False,
                },
            },
            "success_conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "object": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["object", "target"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["task_summary", "robots", "objects", "plan", "success_conditions"],
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

    def plan_gridworld_task(
        self,
        instruction: str,
        num_robots: int,
        layout_name: str,
        available_locations: List[str],
        num_circles: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Convert a free-form gridworld scenario into a structured team execution spec.
        """

        cleaned_instruction = instruction.strip()
        if not cleaned_instruction:
            raise ValueError("Instruction must be a non-empty string.")
        if num_robots < 1:
            raise ValueError("num_robots must be at least 1.")
        if num_circles is not None and num_circles < 1:
            raise ValueError("num_circles must be at least 1 when provided.")
        if not available_locations:
            raise ValueError("available_locations must not be empty.")

        max_repair_rounds = max(0, int(os.getenv("GRIDWORLD_REPAIR_ROUNDS", "2")))
        messages = self._build_gridworld_messages(
            instruction=cleaned_instruction,
            num_robots=num_robots,
            layout_name=layout_name,
            available_locations=available_locations,
            num_circles=num_circles,
        )
        last_error: Optional[Exception] = None

        for attempt_index in range(max_repair_rounds + 1):
            response = self._request_completion(
                messages,
                json_schema=GRIDWORLD_PLAN_SCHEMA,
            )
            payload = self._parse_json_response(response)

            try:
                parsed_payload = self._parse_gridworld_spec(
                    payload=payload,
                    instruction=cleaned_instruction,
                    num_robots=num_robots,
                    available_locations=available_locations,
                    num_circles=num_circles,
                )
                normalized_payload = self._normalize_gridworld_payload_for_execution(
                    parsed_payload
                )
                self._validate_gridworld_bt_compatibility(normalized_payload)
                return normalized_payload
            except ValueError as error:
                last_error = error
                if attempt_index >= max_repair_rounds:
                    break

                messages = self._build_gridworld_repair_messages(
                    instruction=cleaned_instruction,
                    num_robots=num_robots,
                    layout_name=layout_name,
                    available_locations=available_locations,
                    num_circles=num_circles,
                    current_payload=payload if isinstance(payload, dict) else {},
                    validation_error=str(error),
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gridworld planning failed before a scenario could be produced.")

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

    def _build_gridworld_messages(
        self,
        instruction: str,
        num_robots: int,
        layout_name: str,
        available_locations: List[str],
        num_circles: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Create the LLM prompt for typed multi-robot gridworld scenarios.
        """

        robot_names = ["robot_{}".format(index) for index in range(1, num_robots + 1)]
        user_lines = [
            'Scenario: "{}"'.format(instruction),
            "Layout: {}".format(layout_name),
            "Requested robot count: {}. The robots list must contain exactly {} entries, even if some robots stay idle.".format(
                num_robots,
                num_robots,
            ),
            "Use exactly these robot names: {}".format(", ".join(robot_names)),
        ]
        if num_circles is not None:
            user_lines.append(
                "Requested object count: {}. Create exactly {} movable task objects unless the instruction explicitly overrides the object count.".format(
                    num_circles,
                    num_circles,
                )
            )
        user_lines.extend(
            [
                "Available symbolic locations: {}".format(
                    ", ".join(sorted(available_locations))
                ),
                "Return JSON with keys task_summary, robots, objects, plan, success_conditions.",
            ]
        )
        return [
            {
                "role": "system",
                "content": GRIDWORLD_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "\n".join(user_lines),
            },
        ]

    def _build_gridworld_repair_messages(
        self,
        instruction: str,
        num_robots: int,
        layout_name: str,
        available_locations: List[str],
        num_circles: Optional[int],
        current_payload: Dict[str, Any],
        validation_error: str,
    ) -> List[Dict[str, str]]:
        """
        Ask the model to repair an invalid gridworld scenario spec.
        """

        robot_names = self._expected_gridworld_robot_names(num_robots)
        user_lines = [
            'Original scenario: "{}"'.format(instruction),
            "Layout: {}".format(layout_name),
            "Requested robot count: {}. The robots list must contain exactly {} entries, even if some robots stay idle.".format(
                num_robots,
                num_robots,
            ),
            "Use exactly these robot names: {}".format(", ".join(robot_names)),
        ]
        if num_circles is not None:
            user_lines.append(
                "Requested object count: {}. Keep exactly {} movable task objects unless the original instruction explicitly overrides the count.".format(
                    num_circles,
                    num_circles,
                )
            )
        user_lines.extend(
            [
                "Available symbolic locations: {}".format(
                    ", ".join(sorted(available_locations))
                ),
                "Current invalid JSON:",
                json.dumps(current_payload, indent=2),
                "Validation error:",
                validation_error,
                "Return corrected JSON with keys task_summary, robots, objects, plan, success_conditions.",
            ]
        )
        return [
            {
                "role": "system",
                "content": "{}\n\n{}".format(
                    GRIDWORLD_SYSTEM_PROMPT,
                    GRIDWORLD_REPAIR_PROMPT,
                ),
            },
            {
                "role": "user",
                "content": "\n".join(user_lines),
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

    def _parse_gridworld_spec(
        self,
        payload: Any,
        instruction: str,
        num_robots: int,
        available_locations: List[str],
        num_circles: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate and normalize a structured gridworld scenario spec.
        """

        if not isinstance(payload, dict):
            raise ValueError("Gridworld planning response must be a JSON object.")

        allowed_locations = {location.strip() for location in available_locations if location.strip()}
        if not allowed_locations:
            raise ValueError("Gridworld planning requires at least one available location.")

        task_summary = payload.get("task_summary")
        if not isinstance(task_summary, str) or not task_summary.strip():
            task_summary = instruction

        raw_plan = payload.get("plan")
        if not isinstance(raw_plan, list) or not raw_plan:
            raise ValueError("Gridworld planning response must include a non-empty 'plan' list.")

        expected_robot_names = self._expected_gridworld_robot_names(num_robots)
        plan = self._canonicalize_plan(raw_plan)
        robot_name_mapping = self._build_gridworld_robot_name_mapping(
            raw_robots=payload.get("robots"),
            plan=plan,
            expected_robot_names=expected_robot_names,
        )
        plan = self._normalize_gridworld_plan_robot_references(
            plan=plan,
            expected_robot_names=expected_robot_names,
            robot_name_mapping=robot_name_mapping,
        )
        robots = self._normalize_gridworld_robot_specs(
            raw_robots=payload.get("robots"),
            expected_robot_names=expected_robot_names,
            allowed_locations=allowed_locations,
            plan=plan,
            robot_name_mapping=robot_name_mapping,
        )

        raw_objects = payload.get("objects")
        if not isinstance(raw_objects, list) or not raw_objects:
            raise ValueError("Gridworld planning response must include a non-empty 'objects' list.")

        objects: List[Dict[str, str]] = []
        seen_object_names = set()
        for raw_object in raw_objects:
            if not isinstance(raw_object, dict):
                raise ValueError("Each gridworld object spec must be a JSON object.")

            object_name = self._first_non_empty_string(raw_object.get("name"))
            if object_name is None:
                raise ValueError("Each gridworld object spec must include a non-empty 'name'.")
            if object_name in seen_object_names:
                raise ValueError("Gridworld object names must be unique.")

            raw_kind = self._first_non_empty_string(raw_object.get("kind"))
            raw_shape = self._first_non_empty_string(raw_object.get("shape"))
            kind = infer_object_kind(object_name, explicit_kind=raw_kind, shape=raw_shape)
            if kind not in SUPPORTED_OBJECT_KINDS:
                raise ValueError(
                    "Gridworld object '{}' must use a supported kind.".format(object_name)
                )
            shape = (raw_shape or default_render_shape(kind)).strip().lower()
            if shape not in SUPPORTED_RENDER_SHAPES:
                raise ValueError(
                    "Gridworld object '{}' must use a supported shape.".format(object_name)
                )

            objects.append({"name": object_name, "kind": kind, "shape": shape})
            seen_object_names.add(object_name)

        if num_circles is not None and len(objects) != num_circles:
            raise ValueError(
                "Gridworld planner must describe exactly {} objects, but returned {}.".format(
                    num_circles,
                    len(objects),
                )
            )

        self._validate_gridworld_plan(
            plan=plan,
            robot_specs=robots,
            object_specs=objects,
            available_locations=allowed_locations,
        )

        raw_success_conditions = payload.get("success_conditions")
        if not isinstance(raw_success_conditions, list):
            raise ValueError(
                "Gridworld planning response must include a 'success_conditions' list."
            )

        success_conditions: List[Dict[str, str]] = []
        for raw_condition in raw_success_conditions:
            if not isinstance(raw_condition, dict):
                raise ValueError("Each success condition must be a JSON object.")

            object_name = self._first_non_empty_string(raw_condition.get("object"))
            target = self._first_non_empty_string(raw_condition.get("target"))
            if object_name is None or object_name not in seen_object_names:
                raise ValueError("Each success condition must reference a known object.")
            if target is None or target not in allowed_locations:
                raise ValueError("Each success condition must use a supported target.")

            success_conditions.append({"object": object_name, "target": target})

        if not success_conditions:
            inferred_conditions = []
            for step in plan:
                action_name = self._canonicalize_action_name(step.get("action", ""))
                if action_name not in {"Place", "Insert"}:
                    continue
                object_name = step.get("object")
                target = step.get("target")
                if object_name and target:
                    inferred_conditions.append({"object": object_name, "target": target})

            if not inferred_conditions:
                raise ValueError(
                    "Gridworld planning response must include at least one success condition."
                )
            success_conditions = inferred_conditions

        return {
            "task_summary": task_summary.strip(),
            "robots": robots,
            "objects": objects,
            "plan": plan,
            "success_conditions": success_conditions,
        }

    def _expected_gridworld_robot_names(self, num_robots: int) -> List[str]:
        """
        Return the exact robot identifiers a gridworld plan should use.
        """

        return ["robot_{}".format(index) for index in range(1, num_robots + 1)]

    def _canonicalize_gridworld_robot_name(
        self,
        raw_name: Any,
        expected_robot_names: Sequence[str],
    ) -> Optional[str]:
        """
        Map loose robot labels like ``Robot 4`` or ``robot4`` onto the exact
        ``robot_4`` form expected by the simulator.
        """

        if not isinstance(raw_name, str) or not raw_name.strip():
            return None

        cleaned_name = raw_name.strip()
        lowered_name = cleaned_name.lower()
        expected_set = set(expected_robot_names)
        if lowered_name in expected_set:
            return lowered_name

        match = re.fullmatch(r"robot[\s_-]*(\d+)", cleaned_name, flags=re.IGNORECASE)
        if not match:
            return None

        canonical_name = "robot_{}".format(int(match.group(1)))
        if canonical_name not in expected_set:
            return None
        return canonical_name

    def _build_gridworld_robot_name_mapping(
        self,
        raw_robots: Any,
        plan: Sequence[Dict[str, str]],
        expected_robot_names: Sequence[str],
    ) -> Dict[str, str]:
        """
        Build a best-effort mapping from loose LLM robot labels onto the exact
        requested robot ids.
        """

        mapping: Dict[str, str] = {}
        used_expected_names = set()
        pending_names: List[str] = []

        def register_name(raw_name: Optional[str]) -> None:
            if raw_name is None or raw_name in mapping:
                return

            canonical_name = self._canonicalize_gridworld_robot_name(
                raw_name,
                expected_robot_names,
            )
            if canonical_name is not None:
                mapping[raw_name] = canonical_name
                used_expected_names.add(canonical_name)
                return

            pending_names.append(raw_name)

        if isinstance(raw_robots, list):
            for raw_robot in raw_robots:
                if not isinstance(raw_robot, dict):
                    continue
                register_name(
                    self._first_non_empty_string(
                        raw_robot.get("name"),
                        raw_robot.get("robot"),
                    )
                )

        for step in plan:
            register_name(self._first_non_empty_string(step.get("robot")))
            register_name(self._first_non_empty_string(step.get("recipient")))

        remaining_expected_names = [
            name for name in expected_robot_names if name not in used_expected_names
        ]
        for pending_name in pending_names:
            if not remaining_expected_names:
                break
            mapping[pending_name] = remaining_expected_names.pop(0)

        return mapping

    def _normalize_gridworld_plan_robot_references(
        self,
        plan: Sequence[Dict[str, str]],
        expected_robot_names: Sequence[str],
        robot_name_mapping: Dict[str, str],
    ) -> List[Dict[str, str]]:
        """
        Rewrite robot and recipient fields to the exact simulator robot ids.
        """

        normalized_plan: List[Dict[str, str]] = []
        for step in plan:
            normalized_step = dict(step)
            for field_name in ("robot", "recipient"):
                raw_name = self._first_non_empty_string(normalized_step.get(field_name))
                if raw_name is None:
                    continue
                normalized_step[field_name] = (
                    robot_name_mapping.get(raw_name)
                    or self._canonicalize_gridworld_robot_name(
                        raw_name,
                        expected_robot_names,
                    )
                    or raw_name
                )
            normalized_plan.append(normalized_step)
        return normalized_plan

    def _normalize_gridworld_robot_specs(
        self,
        raw_robots: Any,
        expected_robot_names: Sequence[str],
        allowed_locations: set[str],
        plan: Sequence[Dict[str, str]],
        robot_name_mapping: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Normalize or synthesize robot specs so the payload always contains the
        exact requested robot set.
        """

        normalized_robots: List[Dict[str, Any]] = []
        used_start_locations = set()
        raw_robot_lookup: Dict[str, Dict[str, Any]] = {}

        if isinstance(raw_robots, list):
            for raw_robot in raw_robots:
                if not isinstance(raw_robot, dict):
                    continue
                raw_name = self._first_non_empty_string(
                    raw_robot.get("name"),
                    raw_robot.get("robot"),
                )
                if raw_name is None:
                    continue
                normalized_name = (
                    robot_name_mapping.get(raw_name)
                    or self._canonicalize_gridworld_robot_name(
                        raw_name,
                        expected_robot_names,
                    )
                )
                if normalized_name is None or normalized_name in raw_robot_lookup:
                    continue
                raw_robot_lookup[normalized_name] = dict(raw_robot)

        for robot_name in expected_robot_names:
            normalized_robot = self._coerce_gridworld_robot_spec(
                robot_name=robot_name,
                raw_robot=raw_robot_lookup.get(robot_name, {}),
                allowed_locations=allowed_locations,
                used_start_locations=used_start_locations,
                plan=plan,
            )
            normalized_robots.append(normalized_robot)
            used_start_locations.add(normalized_robot["start_location"])

        if len(normalized_robots) != len(expected_robot_names):
            raise ValueError(
                "Gridworld planner must describe exactly {} robots, but returned {}.".format(
                    len(expected_robot_names),
                    len(normalized_robots),
                )
            )
        return normalized_robots

    def _coerce_gridworld_robot_spec(
        self,
        robot_name: str,
        raw_robot: Dict[str, Any],
        allowed_locations: set[str],
        used_start_locations: set[str],
        plan: Sequence[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Coerce one robot spec into the subset the simulator needs, filling in
        sane defaults when the model omits a field.
        """

        can_move = raw_robot.get("can_move")
        if not isinstance(can_move, bool):
            can_move = True

        role = self._first_non_empty_string(raw_robot.get("role"))
        if role is None:
            role = "worker" if can_move else "waiting_observer"

        start_location = self._first_non_empty_string(raw_robot.get("start_location"))
        if start_location not in allowed_locations:
            start_location = self._infer_gridworld_robot_start_location(
                robot_name=robot_name,
                plan=plan,
                allowed_locations=allowed_locations,
                used_start_locations=used_start_locations,
            )

        return {
            "name": robot_name,
            "role": role,
            "start_location": start_location,
            "can_move": can_move,
        }

    def _infer_gridworld_robot_start_location(
        self,
        robot_name: str,
        plan: Sequence[Dict[str, str]],
        allowed_locations: set[str],
        used_start_locations: set[str],
    ) -> str:
        """
        Infer a reasonable fallback start location for a repaired robot spec.
        """

        for step in plan:
            step_robot = self._first_non_empty_string(step.get("robot"))
            if step_robot == robot_name:
                for field_name in ("target", "location"):
                    candidate = self._first_non_empty_string(step.get(field_name))
                    if candidate in allowed_locations:
                        return candidate

            recipient_name = self._first_non_empty_string(step.get("recipient"))
            if recipient_name == robot_name:
                candidate = self._first_non_empty_string(step.get("location"))
                if candidate in allowed_locations:
                    return candidate

        preferred_locations = [
            "center",
            "left_mid",
            "right_mid",
            "upper_mid",
            "lower_mid",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
            "assembly_station",
            "tool_rack",
            "precision_station",
            "chassis",
            "bearing_block",
            "panel_slot",
        ]
        for candidate in preferred_locations:
            if candidate in allowed_locations and candidate not in used_start_locations:
                return candidate

        if allowed_locations:
            return sorted(allowed_locations)[0]
        raise ValueError("Gridworld planning requires at least one available location.")

    def _normalize_gridworld_payload_for_execution(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize common handoff-spec patterns into forms the MRBTP backbone accepts.
        """

        normalized_payload = dict(payload)
        raw_plan = payload.get("plan", [])
        if not isinstance(raw_plan, list):
            return normalized_payload

        terminal_actions = {"Place", "Insert", "Handoff"}
        current_segment: List[Dict[str, str]] = []
        normalized_plan: List[Dict[str, str]] = []

        for step in raw_plan:
            if not isinstance(step, dict):
                continue

            canonical_step = dict(step)
            current_segment.append(canonical_step)
            action_name = self._canonicalize_action_name(canonical_step.get("action", ""))
            if action_name not in terminal_actions:
                continue

            normalized_plan.extend(self._normalize_gridworld_segment(current_segment))
            current_segment = []

        if current_segment:
            normalized_plan.extend(current_segment)

        normalized_payload["plan"] = self._normalize_stationary_receiver_inventory(
            plan=normalized_plan,
            raw_robots=payload.get("robots", []),
            success_conditions=payload.get("success_conditions", []),
        )
        return normalized_payload

    def _normalize_gridworld_segment(
        self,
        segment_steps: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Drop recipient travel hints from handoff segments because the MRBTP runtime
        synthesizes receiver preparation automatically.
        """

        if not segment_steps:
            return []

        terminal_step = dict(segment_steps[-1])
        action_name = self._canonicalize_action_name(terminal_step.get("action", ""))
        if action_name != "Handoff":
            return [dict(step) for step in segment_steps]

        giver_name = terminal_step.get("robot")
        recipient_name = terminal_step.get("recipient")
        location_name = terminal_step.get("location")

        normalized_segment: List[Dict[str, str]] = []
        for step in segment_steps[:-1]:
            step_robot = step.get("robot")
            step_action = self._canonicalize_action_name(step.get("action", ""))
            step_target = step.get("target")

            if (
                step_robot == recipient_name
                and step_action == "MoveTo"
                and location_name
                and step_target == location_name
            ):
                continue

            if step_robot is not None and giver_name is not None and step_robot != giver_name:
                continue

            normalized_segment.append(dict(step))

        normalized_segment.append(terminal_step)
        return normalized_segment

    def _normalize_stationary_receiver_inventory(
        self,
        plan: Sequence[Dict[str, str]],
        raw_robots: Any,
        success_conditions: Any,
    ) -> List[Dict[str, str]]:
        """
        Keep stationary receivers empty-handed between repeated handoffs.

        The gridworld runtime models a single carried object per robot. When the
        model asks a waiting receiver to accept multiple objects in sequence, we
        can safely serialize that flow by inserting a local Place at the waiting
        location before the next incoming Handoff begins.
        """

        stationary_locations: Dict[str, str] = {}
        if isinstance(raw_robots, list):
            for raw_robot in raw_robots:
                if not isinstance(raw_robot, dict):
                    continue

                robot_name = self._first_non_empty_string(raw_robot.get("name"))
                start_location = self._first_non_empty_string(raw_robot.get("start_location"))
                can_move = raw_robot.get("can_move")
                if robot_name is None or start_location is None or not isinstance(can_move, bool):
                    continue
                if not can_move:
                    stationary_locations[robot_name] = start_location

        if not stationary_locations:
            return [dict(step) for step in plan]

        desired_targets: Dict[str, str] = {}
        if isinstance(success_conditions, list):
            for raw_condition in success_conditions:
                if not isinstance(raw_condition, dict):
                    continue
                object_name = self._first_non_empty_string(raw_condition.get("object"))
                target_name = self._first_non_empty_string(raw_condition.get("target"))
                if object_name is None or target_name is None:
                    continue
                desired_targets[object_name] = target_name

        normalized_plan: List[Dict[str, str]] = []
        held_by_stationary: Dict[str, Optional[str]] = {
            robot_name: None for robot_name in stationary_locations
        }

        for step in plan:
            canonical_step = dict(step)
            robot_name = self._first_non_empty_string(canonical_step.get("robot"))
            action_name = self._canonicalize_action_name(canonical_step.get("action", ""))
            object_name = self._first_non_empty_string(canonical_step.get("object"))
            recipient_name = self._first_non_empty_string(canonical_step.get("recipient"))

            for stationary_robot, held_object in list(held_by_stationary.items()):
                if held_object is None or robot_name == stationary_robot:
                    continue

                target_name = desired_targets.get(held_object)
                if target_name != stationary_locations[stationary_robot]:
                    continue

                normalized_plan.append(
                    {
                        "robot": stationary_robot,
                        "action": "Place",
                        "object": held_object,
                        "target": stationary_locations[stationary_robot],
                    }
                )
                held_by_stationary[stationary_robot] = None

            normalized_plan.append(canonical_step)

            if action_name == "Pick" and robot_name in held_by_stationary:
                held_by_stationary[robot_name] = object_name
            elif action_name in {"Place", "Insert"} and robot_name in held_by_stationary:
                if held_by_stationary[robot_name] == object_name:
                    held_by_stationary[robot_name] = None
            elif action_name == "Handoff":
                if robot_name in held_by_stationary and held_by_stationary[robot_name] == object_name:
                    held_by_stationary[robot_name] = None
                if recipient_name in held_by_stationary:
                    held_by_stationary[recipient_name] = object_name

        for robot_name, object_name in held_by_stationary.items():
            if object_name is None:
                continue

            target_name = desired_targets.get(object_name)
            if target_name != stationary_locations[robot_name]:
                continue

            normalized_plan.append(
                {
                    "robot": robot_name,
                    "action": "Place",
                    "object": object_name,
                    "target": stationary_locations[robot_name],
                }
            )
            held_by_stationary[robot_name] = None

        return normalized_plan

    def _validate_gridworld_bt_compatibility(self, payload: Dict[str, Any]) -> None:
        """
        Ensure the normalized gridworld plan compiles through the MRBTP backbone.
        """

        raw_plan = payload.get("plan")
        raw_robots = payload.get("robots")
        if not isinstance(raw_plan, list) or not isinstance(raw_robots, list):
            raise ValueError("Gridworld payload is missing plan or robots for BT validation.")

        plan: List[Dict[str, str]] = []
        for step in raw_plan:
            if isinstance(step, dict):
                plan.append(dict(step))

        profiles = self._build_gridworld_profiles_for_bt(raw_robots, plan)
        try:
            build_multi_robot_tree_from_json(plan, robot_profiles=profiles)
        except ValueError as error:
            raise ValueError("Gridworld BT compatibility check failed: {}".format(error)) from error

    def _build_gridworld_profiles_for_bt(
        self,
        raw_robots: Sequence[Dict[str, Any]],
        plan: Sequence[Dict[str, str]],
    ) -> List[RobotProfile]:
        """
        Derive MRBTP robot profiles from the gridworld spec for compatibility checks.
        """

        action_lookup: Dict[str, set[str]] = {}
        tool_lookup: Dict[str, set[str]] = {}
        for step in plan:
            robot_name = self._first_non_empty_string(step.get("robot"))
            action_name = self._first_non_empty_string(step.get("action"))
            if robot_name is None or action_name is None:
                continue

            action_lookup.setdefault(robot_name, set()).add(
                self._canonicalize_action_name(action_name)
            )
            tool_name = self._first_non_empty_string(step.get("tool"))
            if tool_name is not None:
                tool_lookup.setdefault(robot_name, {DEFAULT_TOOL_NAME}).add(tool_name)

        profiles: List[RobotProfile] = []
        for priority, raw_robot in enumerate(raw_robots):
            robot_name = self._first_non_empty_string(raw_robot.get("name"))
            start_location = self._first_non_empty_string(raw_robot.get("start_location"))
            can_move = raw_robot.get("can_move")
            if robot_name is None or start_location is None or not isinstance(can_move, bool):
                continue

            capabilities = set(action_lookup.get(robot_name, set()))
            if can_move:
                capabilities.add("MoveTo")

            profiles.append(
                RobotProfile(
                    name=robot_name,
                    capabilities=tuple(sorted(capabilities)),
                    start_location=start_location,
                    available_tools=tuple(sorted(tool_lookup.get(robot_name, {DEFAULT_TOOL_NAME}))),
                    default_tool=DEFAULT_TOOL_NAME,
                    priority=priority,
                )
            )

        return profiles

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
        robot_value = self._first_non_empty_string(
            step.get("robot"),
            step.get("agent"),
            step.get("robot_name"),
        )
        if robot_value is not None:
            canonical_step["robot"] = robot_value

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

        if canonical_action_name == "ChangeTool":
            tool_value = self._first_non_empty_string(
                step.get("tool"),
                step.get("target"),
                step.get("object"),
                parsed_arguments[0] if len(parsed_arguments) >= 1 else None,
            )
            if tool_value is not None:
                canonical_step["tool"] = tool_value
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

        if canonical_action_name == "Handoff":
            recipient_value = self._first_non_empty_string(
                step.get("recipient"),
                step.get("to"),
                step.get("target"),
                parsed_arguments[1] if len(parsed_arguments) >= 2 else None,
            )
            location_value = self._first_non_empty_string(
                step.get("location"),
                step.get("meeting_point"),
                parsed_arguments[2] if len(parsed_arguments) >= 3 else None,
            )
            if object_value is not None:
                canonical_step["object"] = object_value
            if recipient_value is not None:
                canonical_step["recipient"] = recipient_value
            if location_value is not None:
                canonical_step["location"] = location_value
            return canonical_step

        target_value = self._first_non_empty_string(
            step.get("target"),
            step.get("destination"),
            step.get("location"),
            parsed_arguments[1] if len(parsed_arguments) >= 2 else None,
        )
        tool_value = self._first_non_empty_string(
            step.get("tool"),
            parsed_arguments[2] if len(parsed_arguments) >= 3 else None,
        )

        if object_value is not None:
            canonical_step["object"] = object_value
        if target_value is not None:
            canonical_step["target"] = target_value
        if tool_value is not None:
            canonical_step["tool"] = tool_value
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
            "changetool": "ChangeTool",
            "handoff": "Handoff",
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

            canonical_action_name = self._canonicalize_action_name(action_name)
            object_name = step.get("object")
            target = step.get("target")
            tool = step.get("tool")
            recipient = step.get("recipient")
            location = step.get("location")

            if canonical_action_name in {"Pick", "Place", "Insert", "Handoff"}:
                if not isinstance(object_name, str) or not object_name.strip():
                    raise ValueError(
                        "Plan step {} ({}) must include a non-empty 'object'.".format(
                            index, canonical_action_name
                        )
                    )

            if canonical_action_name in {"MoveTo", "Place", "Insert"}:
                if not isinstance(target, str) or not target.strip():
                    raise ValueError(
                        "Plan step {} ({}) must include a non-empty 'target'.".format(
                            index, canonical_action_name
                        )
                    )

            if canonical_action_name == "ChangeTool":
                if not isinstance(tool, str) or not tool.strip():
                    raise ValueError(
                        "Plan step {} (ChangeTool) must include a non-empty 'tool'.".format(
                            index
                        )
                    )

            if canonical_action_name == "Handoff":
                if not isinstance(recipient, str) or not recipient.strip():
                    raise ValueError(
                        "Plan step {} (Handoff) must include a non-empty 'recipient'.".format(
                            index
                        )
                    )
                if not isinstance(location, str) or not location.strip():
                    raise ValueError(
                        "Plan step {} (Handoff) must include a non-empty 'location'.".format(
                            index
                        )
                    )

    def _validate_gridworld_plan(
        self,
        plan: List[Dict[str, str]],
        robot_specs: List[Dict[str, Any]],
        object_specs: List[Dict[str, str]],
        available_locations: set[str],
    ) -> None:
        """
        Validate robot-specific gridworld plans produced by the LLM.
        """

        self._validate_plan(plan)

        robot_lookup = {robot["name"]: robot for robot in robot_specs}
        object_lookup = {item["name"]: item for item in object_specs}
        object_names = set(object_lookup)

        for index, step in enumerate(plan, start=1):
            robot_name = step.get("robot")
            if not isinstance(robot_name, str) or robot_name not in robot_lookup:
                raise ValueError(
                    "Gridworld plan step {} must reference a known robot.".format(index)
                )

            robot_spec = robot_lookup[robot_name]
            action_name = self._canonicalize_action_name(step.get("action", ""))
            object_name = step.get("object")
            target = step.get("target")
            location = step.get("location")
            recipient = step.get("recipient")

            if object_name is not None and object_name not in object_names:
                raise ValueError(
                    "Gridworld plan step {} references unknown object '{}'.".format(
                        index,
                        object_name,
                    )
                )

            for label in (target, location):
                if label is not None and label not in available_locations:
                    raise ValueError(
                        "Gridworld plan step {} uses unsupported location '{}'.".format(
                            index,
                            label,
                        )
                    )

            if recipient is not None and recipient not in robot_lookup:
                raise ValueError(
                    "Gridworld plan step {} references unknown recipient '{}'.".format(
                        index,
                        recipient,
                    )
                )

            if action_name == "Insert" and object_name is not None and target is not None:
                object_kind = str(
                    object_lookup[object_name].get(
                        "kind",
                        object_lookup[object_name].get("shape", ""),
                    )
                )
                if not is_insert_compatible(object_kind, target):
                    allowed_targets = insert_targets_for_kind(object_kind)
                    raise ValueError(
                        "Gridworld object '{}' of kind '{}' cannot be inserted into '{}'. Allowed targets: {}.".format(
                            object_name,
                            object_kind,
                            target,
                            ", ".join(sorted(allowed_targets)),
                        )
                    )

            if not robot_spec["can_move"]:
                if action_name == "MoveTo":
                    raise ValueError(
                        "Gridworld robot '{}' is marked can_move=false but plan step {} uses MoveTo.".format(
                            robot_name,
                            index,
                        )
                    )
                required_location = location or target
                if required_location is not None and required_location != robot_spec["start_location"]:
                    raise ValueError(
                        "Gridworld robot '{}' is stationary, so step {} must stay at '{}'.".format(
                            robot_name,
                            index,
                            robot_spec["start_location"],
                        )
                    )
