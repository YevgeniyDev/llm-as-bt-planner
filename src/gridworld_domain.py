"""
Shared semantic definitions for the gridworld simulator.
"""

from __future__ import annotations

from typing import FrozenSet, Optional


DEFAULT_TOOL_NAME = "default_gripper"
SUPPORTED_RENDER_SHAPES: FrozenSet[str] = frozenset({"circle", "square", "triangle"})
SUPPORTED_OBJECT_KINDS: FrozenSet[str] = frozenset(
    {
        "circle",
        "square",
        "triangle",
        "gear",
        "shaft",
        "pin",
        "plate",
    }
)
DEFAULT_RENDER_SHAPES = {
    "circle": "circle",
    "square": "square",
    "triangle": "triangle",
    "gear": "circle",
    "shaft": "square",
    "pin": "triangle",
    "plate": "square",
}
TOOL_STATION_LOCATIONS: FrozenSet[str] = frozenset({"tool_rack", "precision_station"})
FIXTURE_LOCATIONS: FrozenSet[str] = frozenset(
    {
        "chassis",
        "bearing_block",
        "panel_slot",
        "assembly_station",
    }
)
INSERT_COMPATIBILITY = {
    "gear": frozenset({"chassis"}),
    "shaft": frozenset({"bearing_block"}),
    "pin": frozenset({"panel_slot"}),
    "plate": frozenset({"assembly_station"}),
}


def infer_object_kind(
    name: str,
    explicit_kind: Optional[str] = None,
    shape: Optional[str] = None,
) -> str:
    if isinstance(explicit_kind, str) and explicit_kind.strip():
        return explicit_kind.strip().lower()

    cleaned_name = name.strip().lower()
    base_name = cleaned_name.split("_", 1)[0]
    if base_name in SUPPORTED_OBJECT_KINDS:
        return base_name

    if isinstance(shape, str) and shape.strip():
        return shape.strip().lower()

    return cleaned_name


def default_render_shape(object_kind: str) -> str:
    cleaned_kind = object_kind.strip().lower()
    return DEFAULT_RENDER_SHAPES.get(cleaned_kind, "circle")


def insert_targets_for_kind(object_kind: str) -> FrozenSet[str]:
    return INSERT_COMPATIBILITY.get(object_kind.strip().lower(), frozenset())


def is_insert_compatible(object_kind: str, target_name: str) -> bool:
    allowed_targets = insert_targets_for_kind(object_kind)
    if not allowed_targets:
        return True
    return target_name.strip() in allowed_targets

