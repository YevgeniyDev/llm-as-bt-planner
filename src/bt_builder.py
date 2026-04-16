"""
Utilities for translating symbolic JSON plans into reactive `py_trees`.
"""

import json
from typing import Any, Dict, List, Optional, Union

import py_trees

from .robot_actions import (
    AtLocation,
    ChangeTool,
    Holding,
    Insert,
    InsertedAt,
    MoveTo,
    ObjectAt,
    Pick,
    Place,
    RobotWorldState,
    ToolEquipped,
)


def build_tree_from_json(
    plan_json: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    world_state: Optional[RobotWorldState] = None,
) -> py_trees.trees.BehaviourTree:
    """
    Build an executable reactive behavior tree from a JSON plan.

    The root sequence uses memory=False so earlier conditions are re-evaluated
    on every tick. Each symbolic step is compiled into a fallback-style subtree
    with explicit state checks, which makes the resulting BT reactive rather
    than a brittle linear script.
    """

    plan = _normalize_plan(plan_json)
    shared_world_state = world_state or RobotWorldState.from_plan(plan)

    root = py_trees.composites.Sequence(name="ReactiveTaskPlan", memory=False)
    children = [
        _create_reactive_subtree(plan, step, position, shared_world_state)
        for position, step in enumerate(plan)
    ]
    root.add_children(children)

    tree = py_trees.trees.BehaviourTree(root=root)
    tree.world_state = shared_world_state  # type: ignore[attr-defined]
    return tree


def _normalize_plan(plan_json: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Accept either a JSON string, a wrapped JSON object, or the raw step list.
    """

    payload: Any = plan_json
    if isinstance(plan_json, str):
        payload = json.loads(plan_json)

    if isinstance(payload, dict):
        if "plan" in payload:
            payload = payload["plan"]
        elif "steps" in payload:
            payload = payload["steps"]

    if not isinstance(payload, list) or not payload:
        raise ValueError("Plan JSON must resolve to a non-empty list of steps.")

    return payload


def _create_reactive_subtree(
    plan: List[Dict[str, Any]],
    step: Dict[str, Any],
    position: int,
    world_state: RobotWorldState,
) -> py_trees.behaviour.Behaviour:
    """
    Compile one symbolic step into a reactive subtree with state checks.
    """

    step_number = position + 1
    raw_action_name = step.get("action")
    if not isinstance(raw_action_name, str):
        raise ValueError("Plan step {} is missing the 'action' field.".format(step_number))

    action_key = _normalize_action_name(raw_action_name)

    if action_key == "pick":
        object_name = _get_required_field(step, step_number, "object", "item")
        selector_children = _optional_downstream_goal_children(plan, position, world_state, object_name)
        selector_children.extend(
            [
                Holding(object_name, world_state),
                Pick(object_name, world_state),
            ]
        )
        return _fallback("Step {} Pick {}".format(step_number, object_name), selector_children)

    if action_key == "moveto":
        target = _get_required_field(step, step_number, "target", "destination", "location")
        selector_children = _optional_downstream_goal_children(plan, position, world_state)
        selector_children.extend(
            [
                AtLocation(target, world_state),
                MoveTo(target, world_state),
            ]
        )
        return _fallback("Step {} MoveTo {}".format(step_number, target), selector_children)

    if action_key == "changetool":
        tool_name = _get_required_field(step, step_number, "tool", "target", "object")
        return _fallback(
            "Step {} ChangeTool {}".format(step_number, tool_name),
            [
                ToolEquipped(tool_name, world_state),
                ChangeTool(tool_name, world_state),
            ],
        )

    if action_key == "place":
        object_name = _get_required_field(step, step_number, "object", "item")
        target = _get_required_field(step, step_number, "target", "destination", "location")
        return _fallback(
            "Step {} Place {}".format(step_number, object_name),
            [
                ObjectAt(object_name, target, world_state),
                _sequence(
                    "RecoverThenPlace({})".format(object_name),
                    [
                        _fallback(
                            "EnsureHolding({})".format(object_name),
                            [
                                Holding(object_name, world_state),
                                Pick(object_name, world_state),
                            ],
                        ),
                        _fallback(
                            "EnsureAt({})".format(target),
                            [
                                AtLocation(target, world_state),
                                MoveTo(target, world_state),
                            ],
                        ),
                        Place(object_name, target, world_state),
                    ],
                ),
            ],
        )

    if action_key == "insert":
        object_name = _get_required_field(step, step_number, "object", "item")
        target = _get_required_field(step, step_number, "target", "destination", "location")
        required_tool = _get_optional_field(step, "tool")
        recovery_children: List[py_trees.behaviour.Behaviour] = [
            _fallback(
                "EnsureHolding({})".format(object_name),
                [
                    Holding(object_name, world_state),
                    Pick(object_name, world_state),
                ],
            ),
        ]
        if required_tool:
            recovery_children.append(
                _fallback(
                    "EnsureTool({})".format(required_tool),
                    [
                        ToolEquipped(required_tool, world_state),
                        ChangeTool(required_tool, world_state),
                    ],
                )
            )
        recovery_children.extend(
            [
                _fallback(
                    "EnsureAt({})".format(target),
                    [
                        AtLocation(target, world_state),
                        MoveTo(target, world_state),
                    ],
                ),
                Insert(object_name, target, world_state, required_tool=required_tool),
            ]
        )
        return _fallback(
            "Step {} Insert {}".format(step_number, object_name),
            [
                InsertedAt(object_name, target, world_state),
                _sequence(
                    "RecoverThenInsert({})".format(object_name),
                    recovery_children,
                ),
            ],
        )

    if action_key == "handoff":
        raise ValueError(
            "Unsupported action '{}' at plan step {}. Use multi-robot mode for Handoff."
            .format(raw_action_name, step_number)
        )

    raise ValueError(
        "Unsupported action '{}' at plan step {}.".format(raw_action_name, step_number)
    )


def _optional_downstream_goal_children(
    plan: List[Dict[str, Any]],
    position: int,
    world_state: RobotWorldState,
    object_name: Optional[str] = None,
) -> List[py_trees.behaviour.Behaviour]:
    """
    Allow earlier steps to skip when a later terminal goal is already satisfied.
    """

    terminal_goal = _find_downstream_terminal_goal(plan, position, world_state, object_name)
    if terminal_goal is None:
        return []
    return [terminal_goal]


def _find_downstream_terminal_goal(
    plan: List[Dict[str, Any]],
    position: int,
    world_state: RobotWorldState,
    object_name: Optional[str] = None,
) -> Optional[py_trees.behaviour.Behaviour]:
    """
    Find the next placement or insertion goal relevant to the current step.
    """

    normalized_object_name = object_name.strip() if object_name else None

    for later_step in plan[position + 1 :]:
        raw_action_name = later_step.get("action")
        if not isinstance(raw_action_name, str):
            continue

        action_key = _normalize_action_name(raw_action_name)
        if action_key not in {"place", "insert"}:
            continue

        later_object_name = _get_optional_field(later_step, "object", "item")
        if normalized_object_name and later_object_name != normalized_object_name:
            continue

        if not later_object_name:
            continue

        target = _get_optional_field(later_step, "target", "destination", "location")
        if not target:
            continue

        if action_key == "insert":
            return InsertedAt(later_object_name, target, world_state)

        return ObjectAt(later_object_name, target, world_state)

    return None


def _fallback(
    name: str,
    children: List[py_trees.behaviour.Behaviour],
) -> py_trees.composites.Selector:
    selector = py_trees.composites.Selector(name=name, memory=False)
    selector.add_children(children)
    return selector


def _sequence(
    name: str,
    children: List[py_trees.behaviour.Behaviour],
) -> py_trees.composites.Sequence:
    sequence = py_trees.composites.Sequence(name=name, memory=False)
    sequence.add_children(children)
    return sequence


def _normalize_action_name(action_name: str) -> str:
    """Collapse superficial formatting differences in action labels."""

    return action_name.strip().replace("_", "").replace(" ", "").lower()


def _get_optional_field(step: Dict[str, Any], *field_names: str) -> Optional[str]:
    """
    Retrieve the first non-empty string value from the candidate field names.
    """

    for field_name in field_names:
        value = step.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _get_required_field(step: Dict[str, Any], index: int, *field_names: str) -> str:
    """
    Retrieve the first non-empty string value from the candidate field names.
    """

    value = _get_optional_field(step, *field_names)
    if value is not None:
        return value

    raise ValueError(
        "Plan step {} is missing one of the required fields: {}.".format(
            index, ", ".join(field_names)
        )
    )
