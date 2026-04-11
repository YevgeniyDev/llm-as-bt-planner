"""
Utilities for translating symbolic JSON plans into `py_trees` structures.
"""

import json
from typing import Any, Dict, List, Union

import py_trees

from .robot_actions import Insert, MoveTo, Pick, Place


def build_tree_from_json(plan_json: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> py_trees.trees.BehaviourTree:
    """
    Build an executable behavior tree from a JSON plan.

    The tree root is a Sequence with memory=True because a task plan is an
    ordered procedure: once a step succeeds, the executor should resume from the
    next step instead of re-running earlier actions on every tick.
    """

    plan = _normalize_plan(plan_json)

    root = py_trees.composites.Sequence(name="TaskPlanSequence", memory=True)
    children = [_create_action_node(step, index) for index, step in enumerate(plan, start=1)]
    root.add_children(children)

    return py_trees.trees.BehaviourTree(root=root)


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


def _create_action_node(step: Dict[str, Any], index: int) -> py_trees.behaviour.Behaviour:
    """
    Map a symbolic action dictionary to a concrete behavior node.
    """

    raw_action_name = step.get("action")
    if not isinstance(raw_action_name, str):
        raise ValueError("Plan step {} is missing the 'action' field.".format(index))

    action_key = _normalize_action_name(raw_action_name)

    if action_key == "pick":
        return Pick(object_name=_get_required_field(step, index, "object", "item"))

    if action_key == "place":
        return Place(
            object_name=_get_required_field(step, index, "object", "item"),
            target=_get_required_field(step, index, "target", "destination", "location"),
        )

    if action_key == "moveto":
        return MoveTo(target=_get_required_field(step, index, "target", "destination", "location"))

    if action_key == "insert":
        return Insert(
            object_name=_get_required_field(step, index, "object", "item"),
            target=_get_required_field(step, index, "target", "destination", "location"),
        )

    raise ValueError(
        "Unsupported action '{}' at plan step {}.".format(raw_action_name, index)
    )


def _normalize_action_name(action_name: str) -> str:
    """Collapse superficial formatting differences in action labels."""

    return action_name.strip().replace("_", "").replace(" ", "").lower()


def _get_required_field(step: Dict[str, Any], index: int, *field_names: str) -> str:
    """
    Retrieve the first non-empty string value from the candidate field names.
    """

    for field_name in field_names:
        value = step.get(field_name)
        if isinstance(value, str) and value.strip():
            return value

    raise ValueError(
        "Plan step {} is missing one of the required fields: {}.".format(
            index, ", ".join(field_names)
        )
    )
