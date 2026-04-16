"""
Small layout helpers for the local MRBTP-inspired gridworld test harness.

These functions intentionally fill the role that `custom_env.py` had in the
paper's older MiniGrid examples: they let you define reusable room layouts in
plain Python without needing to edit the simulator core.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple


Position = Tuple[int, int]


@dataclass(frozen=True)
class GridLayout:
    name: str
    width: int
    height: int
    walls: FrozenSet[Position]
    preferred_robot_spawns: Tuple[Position, ...]
    preferred_object_spawns: Tuple[Position, ...]


def boundary_walls(width: int, height: int) -> FrozenSet[Position]:
    walls = set()
    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))
    return frozenset(walls)


def build_open_room_layout(width: int = 11, height: int = 11) -> GridLayout:
    walls = set(boundary_walls(width, height))
    robot_spawns = (
        (1, 1),
        (1, 3),
        (1, 5),
        (2, 2),
        (2, 4),
        (2, 6),
    )
    object_spawns = (
        (width - 3, 1),
        (width - 3, 3),
        (width - 3, 5),
        (width - 4, 2),
        (width - 4, 4),
        (width - 4, 6),
    )
    return GridLayout(
        name="open_room",
        width=width,
        height=height,
        walls=frozenset(walls),
        preferred_robot_spawns=robot_spawns,
        preferred_object_spawns=object_spawns,
    )


def build_split_room_layout(width: int = 13, height: int = 13) -> GridLayout:
    walls = set(boundary_walls(width, height))
    center_x = width // 2
    center_y = height // 2

    for y in range(1, height - 1):
        if y not in {2, center_y, height - 3}:
            walls.add((center_x, y))

    for x in range(1, center_x):
        if x != center_x - 1:
            walls.add((x, center_y))

    robot_spawns = (
        (1, 1),
        (2, 2),
        (3, 3),
        (1, center_y - 1),
        (2, center_y - 2),
    )
    object_spawns = (
        (center_x - 2, height - 3),
        (center_x - 3, height - 4),
        (width - 3, height - 3),
        (width - 4, height - 4),
        (width - 3, 2),
    )
    return GridLayout(
        name="split_room",
        width=width,
        height=height,
        walls=frozenset(walls),
        preferred_robot_spawns=robot_spawns,
        preferred_object_spawns=object_spawns,
    )


def build_four_rooms_layout(width: int = 15, height: int = 15) -> GridLayout:
    walls = set(boundary_walls(width, height))
    center_x = width // 2
    center_y = height // 2

    for y in range(1, height - 1):
        if y not in {2, height - 3, center_y - 1, center_y, center_y + 1}:
            walls.add((center_x, y))

    for x in range(1, width - 1):
        if x not in {2, width - 3, center_x - 1, center_x, center_x + 1}:
            walls.add((x, center_y))

    robot_spawns = (
        (1, 1),
        (width - 2, 1),
        (1, height - 2),
        (width - 2, height - 2),
        (2, center_y - 1),
        (width - 3, center_y + 1),
    )
    object_spawns = (
        (3, 3),
        (width - 4, 3),
        (3, height - 4),
        (width - 4, height - 4),
        (center_x - 1, 2),
        (center_x + 1, height - 3),
    )
    return GridLayout(
        name="four_rooms",
        width=width,
        height=height,
        walls=frozenset(walls),
        preferred_robot_spawns=robot_spawns,
        preferred_object_spawns=object_spawns,
    )


def build_handoff_hall_layout(width: int = 15, height: int = 11) -> GridLayout:
    walls = set(boundary_walls(width, height))
    hall_x = width // 2
    hall_y = height // 2

    for y in range(1, height - 1):
        if y not in {hall_y - 1, hall_y, hall_y + 1}:
            walls.add((hall_x, y))

    for x in range(1, hall_x):
        if x not in {hall_x - 2, hall_x - 1}:
            walls.add((x, hall_y + 1))

    robot_spawns = (
        (1, 1),
        (2, 2),
        (1, height - 2),
        (hall_x + 1, height // 2),
        (width - 3, 2),
        (width - 3, height - 2),
    )
    object_spawns = (
        (hall_x - 2, 1),
        (hall_x - 3, 2),
        (hall_x - 2, height - 2),
        (width - 3, 1),
        (width - 4, 2),
        (width - 3, height - 2),
    )
    return GridLayout(
        name="handoff_hall",
        width=width,
        height=height,
        walls=frozenset(walls),
        preferred_robot_spawns=robot_spawns,
        preferred_object_spawns=object_spawns,
    )


def layout_registry() -> Dict[str, GridLayout]:
    """
    Materialize the built-in layouts.

    Add new builders here when you want custom scenes comparable to the paper's
    hand-authored grid layouts.
    """

    open_room = build_open_room_layout()
    split_room = build_split_room_layout()
    four_rooms = build_four_rooms_layout()
    handoff_hall = build_handoff_hall_layout()
    return {
        open_room.name: open_room,
        split_room.name: split_room,
        four_rooms.name: four_rooms,
        handoff_hall.name: handoff_hall,
    }
