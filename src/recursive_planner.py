"""
Algorithm 1 style recursive planning helpers.

This module separates the recursive orchestration from the low-level LLM client:
- MakePlan: ask the LLM whether to decompose or emit a primitive plan
- MakeTree: recursively expand ordered subgoals
- PredictState: roll a symbolic world state forward after each subgoal
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .llm_client import LLMTaskPlanner
from .robot_actions import RobotWorldState


@dataclass
class RecursiveTraceNode:
    instruction: str
    depth: int
    mode: str
    reason: str
    predicted_state_before: str
    predicted_state_after: str
    plan: List[Dict[str, str]] = field(default_factory=list)
    children: List["RecursiveTraceNode"] = field(default_factory=list)


class RecursiveBTPlanner:
    """
    Orchestrate recursive BT planning with predicted-state rollouts.
    """

    def __init__(
        self,
        planner: LLMTaskPlanner,
        max_depth: int = 3,
        max_subgoals_per_level: int = 4,
    ) -> None:
        self.planner = planner
        self.max_depth = max(1, max_depth)
        self.max_subgoals_per_level = max(1, max_subgoals_per_level)

    def make_tree(self, instruction: str) -> RecursiveTraceNode:
        """
        Recursively expand an instruction into a flat executable plan plus trace.
        """

        root_state = RobotWorldState()
        trace, _ = self._make_tree(instruction.strip(), root_state, depth=0)
        return trace

    def _make_tree(
        self,
        instruction: str,
        predicted_state: RobotWorldState,
        depth: int,
    ) -> Tuple[RecursiveTraceNode, RobotWorldState]:
        """
        Recursive Algorithm 1 style expansion.
        """

        remaining_depth = max(1, self.max_depth - depth)
        state_before = predicted_state.summary()

        if remaining_depth <= 1:
            primitive_plan = self.planner.plan_task(
                instruction,
                state_summary=state_before,
            )
            predicted_after = predicted_state.clone().apply_plan(primitive_plan)
            return (
                RecursiveTraceNode(
                    instruction=instruction,
                    depth=depth,
                    mode="primitive",
                    reason="Depth limit reached, so the task was grounded directly into primitive actions.",
                    predicted_state_before=state_before,
                    predicted_state_after=predicted_after.summary(),
                    plan=primitive_plan,
                ),
                predicted_after,
            )

        decision = self.planner.choose_recursive_expansion(
            instruction=instruction,
            state_summary=state_before,
            remaining_depth=remaining_depth,
            max_subgoals=self.max_subgoals_per_level,
        )

        if self._should_force_decomposition(instruction, decision.plan, remaining_depth):
            heuristic_subgoals = self._plan_to_subgoal_instructions(decision.plan)
            if len(heuristic_subgoals) > 1:
                decision.kind = "decompose"
                decision.subgoals = heuristic_subgoals
                decision.reason = (
                    "{} {}".format(
                        decision.reason,
                        "The planner heuristically split the long primitive plan into recursive subgoals to preserve Scheme 4 structure.",
                    ).strip()
                )

        if decision.kind == "primitive" or not decision.subgoals:
            primitive_plan = decision.plan or self.planner.plan_task(
                instruction,
                state_summary=state_before,
            )
            predicted_after = predicted_state.clone().apply_plan(primitive_plan)
            return (
                RecursiveTraceNode(
                    instruction=instruction,
                    depth=depth,
                    mode="primitive",
                    reason=decision.reason or "The task was treated as a primitive subproblem.",
                    predicted_state_before=state_before,
                    predicted_state_after=predicted_after.summary(),
                    plan=primitive_plan,
                ),
                predicted_after,
            )

        current_state = predicted_state.clone()
        children: List[RecursiveTraceNode] = []
        flat_plan: List[Dict[str, str]] = []

        for subgoal in decision.subgoals[: self.max_subgoals_per_level]:
            child_trace, current_state = self._make_tree(subgoal, current_state, depth + 1)
            children.append(child_trace)
            flat_plan.extend(child_trace.plan)

        return (
            RecursiveTraceNode(
                instruction=instruction,
                depth=depth,
                mode="decompose",
                reason=decision.reason or "The task was decomposed into ordered subgoals.",
                predicted_state_before=state_before,
                predicted_state_after=current_state.summary(),
                plan=flat_plan,
                children=children,
            ),
            current_state,
        )

    def _should_force_decomposition(
        self,
        instruction: str,
        plan: List[Dict[str, str]],
        remaining_depth: int,
    ) -> bool:
        """
        Keep obviously multi-stage tasks recursive even when the model emits one long plan.
        """

        if remaining_depth <= 1:
            return False

        if len(plan) < 5:
            return False

        object_names = {
            step.get("object")
            for step in plan
            if isinstance(step.get("object"), str) and step.get("object", "").strip()
        }
        if len(object_names) > 1:
            return True

        sentence_count = instruction.count(".") + instruction.count("\n")
        return sentence_count >= 2

    def _plan_to_subgoal_instructions(self, plan: List[Dict[str, str]]) -> List[str]:
        """
        Convert a long primitive plan into higher-level ordered subgoals.
        """

        chunks: List[List[Dict[str, str]]] = []
        current_chunk: List[Dict[str, str]] = []

        for step in plan:
            current_chunk.append(step)
            action_name = step.get("action")
            if not isinstance(action_name, str):
                continue

            action_key = action_name.strip().replace("_", "").replace(" ", "").lower()
            if action_key in {"place", "insert"}:
                chunks.append(current_chunk)
                current_chunk = []

        if current_chunk:
            chunks.append(current_chunk)

        subgoals = []
        for chunk in chunks:
            instruction = self._plan_chunk_to_instruction(chunk)
            if instruction:
                subgoals.append(instruction)

        return subgoals

    def _plan_chunk_to_instruction(self, chunk: List[Dict[str, str]]) -> str:
        """
        Summarize a primitive chunk into a compact natural-language subgoal.
        """

        if not chunk:
            return ""

        last_step = chunk[-1]
        action_name = last_step.get("action")
        if not isinstance(action_name, str):
            return ""

        action_key = action_name.strip().replace("_", "").replace(" ", "").lower()
        object_name = last_step.get("object")
        target = last_step.get("target")

        if action_key == "place" and isinstance(object_name, str) and isinstance(target, str):
            return "Pick up the {} and place it on the {}.".format(object_name, target)

        if action_key == "insert" and isinstance(object_name, str) and isinstance(target, str):
            return "Pick up the {} and insert it into the {}.".format(object_name, target)

        if action_key == "moveto" and isinstance(target, str):
            return "Move to the {}.".format(target)

        if action_key == "pick" and isinstance(object_name, str):
            return "Pick up the {}.".format(object_name)

        return "Complete this subtask: {}".format(json.dumps(chunk))


def render_recursive_trace(node: RecursiveTraceNode, indent: int = 0) -> str:
    """
    Render the recursive planning trace for demos and reports.
    """

    prefix = "  " * indent
    mode_label = "Primitive" if node.mode == "primitive" else "Decompose"
    lines = [
        "{}- Depth {} {}: {}".format(prefix, node.depth, mode_label, node.instruction),
        "{}  reason: {}".format(prefix, node.reason or "n/a"),
        "{}  predict_before: {}".format(prefix, node.predicted_state_before),
        "{}  predict_after: {}".format(prefix, node.predicted_state_after),
    ]

    if node.mode == "primitive":
        lines.append("{}  plan: {}".format(prefix, json.dumps(node.plan)))

    for child in node.children:
        lines.append(render_recursive_trace(child, indent + 1))

    return "\n".join(lines)
