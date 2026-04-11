"""
Pre-execution validation helpers for reactive symbolic plans.
"""

from typing import Any, Dict, List, Optional, Tuple


def validate_reactive_plan(plan: List[Dict[str, Any]]) -> List[str]:
    """
    Detect plan patterns that are likely to oscillate in a memoryless reactive BT.

    The most important hazard for this prototype is a transfer segment that ends
    in `Place` or `Insert(target=X)` while earlier explicit `MoveTo` steps in the
    same segment target some other location `Y`. Because the BT re-evaluates from
    the top on every tick, the earlier `MoveTo(Y)` and the terminal action's
    recovery `EnsureAt(X)` can fight each other forever.
    """

    warnings: List[str] = []
    segment_start = 0

    for terminal_index, step in enumerate(plan):
        action_name = step.get("action")
        if not isinstance(action_name, str):
            continue

        action_key = _normalize_action_name(action_name)
        if action_key not in {"place", "insert"}:
            continue

        warning = _validate_terminal_segment(plan, segment_start, terminal_index)
        if warning is not None:
            warnings.append(warning)

        segment_start = terminal_index + 1

    return warnings


def _validate_terminal_segment(
    plan: List[Dict[str, Any]],
    segment_start: int,
    terminal_index: int,
) -> Optional[str]:
    """
    Validate one contiguous work segment ending in `Place` or `Insert`.
    """

    terminal_step = plan[terminal_index]
    terminal_action = terminal_step.get("action")
    if not isinstance(terminal_action, str):
        return None

    final_target = _get_optional_field(terminal_step, "target", "destination", "location")
    if final_target is None:
        return None

    object_name = _get_optional_field(terminal_step, "object", "item") or "object"
    move_steps: List[Tuple[int, str]] = []

    for step_index in range(segment_start, terminal_index):
        step = plan[step_index]
        action_name = step.get("action")
        if not isinstance(action_name, str):
            continue

        if _normalize_action_name(action_name) != "moveto":
            continue

        move_target = _get_optional_field(step, "target", "destination", "location")
        if move_target is None:
            continue

        move_steps.append((step_index + 1, move_target))

    conflicting_moves = [
        (step_number, target)
        for step_number, target in move_steps
        if target != final_target
    ]
    if not conflicting_moves:
        return None

    move_descriptions = ", ".join(
        "step {} -> {}".format(step_number, target)
        for step_number, target in conflicting_moves
    )

    return (
        "Steps {}-{} end with {}({}, {}), but earlier MoveTo steps target {}. "
        "In this memoryless reactive BT, those earlier navigation checks will keep "
        "re-enforcing their targets while the terminal recovery subtree re-enforces {}, "
        "which can cause oscillation or non-termination.".format(
            segment_start + 1,
            terminal_index + 1,
            _canonical_action_name(terminal_action),
            object_name,
            final_target,
            move_descriptions,
            final_target,
        )
    )


def _normalize_action_name(action_name: str) -> str:
    return action_name.strip().replace("_", "").replace(" ", "").lower()


def _canonical_action_name(action_name: str) -> str:
    mapping = {
        "pick": "Pick",
        "place": "Place",
        "moveto": "MoveTo",
        "insert": "Insert",
    }
    return mapping.get(_normalize_action_name(action_name), action_name.strip())


def _get_optional_field(step: Dict[str, Any], *field_names: str) -> Optional[str]:
    for field_name in field_names:
        value = step.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
