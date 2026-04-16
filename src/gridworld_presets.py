"""
Reusable natural-language presets for the visual gridworld tester.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


CUSTOM_GRIDWORLD_SCENARIO_PLACEHOLDER = (
    "Write the instructions here and set the number of robots and objects, or leave this unchanged to run the default scenario."
)


@dataclass(frozen=True)
class GridWorldPreset:
    name: str
    scenario_text: str
    num_robots: int
    num_circles: int
    layout_name: str


def precision_insert_payload() -> Dict[str, Any]:
    return {
        "task_summary": "One assembler switches tools and inserts a gear into the chassis.",
        "robots": [
            {
                "name": "robot_1",
                "role": "assembler",
                "start_location": "left_mid",
                "can_move": True,
            },
        ],
        "objects": [
            {"name": "gear_1", "kind": "gear", "shape": "circle"},
        ],
        "plan": [
            {"robot": "robot_1", "action": "MoveTo", "target": "tool_rack"},
            {"robot": "robot_1", "action": "ChangeTool", "tool": "precision_gripper"},
            {"robot": "robot_1", "action": "Pick", "object": "gear_1"},
            {"robot": "robot_1", "action": "MoveTo", "target": "chassis"},
            {
                "robot": "robot_1",
                "action": "Insert",
                "object": "gear_1",
                "target": "chassis",
                "tool": "precision_gripper",
            },
        ],
        "success_conditions": [
            {"object": "gear_1", "target": "chassis"},
        ],
    }


def relay_insert_payload() -> Dict[str, Any]:
    return {
        "task_summary": "A relay handoff passes one shaft to a specialist for insertion.",
        "robots": [
            {
                "name": "robot_1",
                "role": "collector",
                "start_location": "left_mid",
                "can_move": True,
            },
            {
                "name": "robot_2",
                "role": "assembler",
                "start_location": "center",
                "can_move": True,
            },
        ],
        "objects": [
            {"name": "shaft_1", "kind": "shaft", "shape": "square"},
        ],
        "plan": [
            {"robot": "robot_1", "action": "Pick", "object": "shaft_1"},
            {"robot": "robot_1", "action": "MoveTo", "target": "center"},
            {
                "robot": "robot_1",
                "action": "Handoff",
                "object": "shaft_1",
                "recipient": "robot_2",
                "location": "center",
            },
            {"robot": "robot_2", "action": "MoveTo", "target": "tool_rack"},
            {"robot": "robot_2", "action": "ChangeTool", "tool": "precision_gripper"},
            {"robot": "robot_2", "action": "MoveTo", "target": "bearing_block"},
            {
                "robot": "robot_2",
                "action": "Insert",
                "object": "shaft_1",
                "target": "bearing_block",
                "tool": "precision_gripper",
            },
        ],
        "success_conditions": [
            {"object": "shaft_1", "target": "bearing_block"},
        ],
    }


def three_robot_assembly_payload() -> Dict[str, Any]:
    return {
        "task_summary": "Two feeders fetch parts in parallel while one assembler prepares and then inserts both parts.",
        "robots": [
            {
                "name": "robot_1",
                "role": "gear_feeder",
                "start_location": "left_mid",
                "can_move": True,
            },
            {
                "name": "robot_2",
                "role": "shaft_feeder",
                "start_location": "lower_mid",
                "can_move": True,
            },
            {
                "name": "robot_3",
                "role": "assembler",
                "start_location": "center",
                "can_move": True,
            },
        ],
        "objects": [
            {"name": "gear_1", "kind": "gear", "shape": "circle"},
            {"name": "shaft_1", "kind": "shaft", "shape": "square"},
        ],
        "plan": [
            {"robot": "robot_1", "action": "Pick", "object": "gear_1"},
            {"robot": "robot_1", "action": "MoveTo", "target": "left_center"},
            {"robot": "robot_1", "action": "Place", "object": "gear_1", "target": "left_center"},
            {"robot": "robot_2", "action": "Pick", "object": "shaft_1"},
            {"robot": "robot_2", "action": "MoveTo", "target": "right_center"},
            {
                "robot": "robot_2",
                "action": "Place",
                "object": "shaft_1",
                "target": "right_center",
            },
            {"robot": "robot_3", "action": "MoveTo", "target": "tool_rack"},
            {
                "robot": "robot_3",
                "action": "ChangeTool",
                "tool": "precision_gripper",
                "target": "tool_rack",
            },
            {"robot": "robot_3", "action": "MoveTo", "target": "left_center"},
            {"robot": "robot_3", "action": "Pick", "object": "gear_1"},
            {"robot": "robot_3", "action": "MoveTo", "target": "chassis"},
            {
                "robot": "robot_3",
                "action": "Insert",
                "object": "gear_1",
                "target": "chassis",
                "tool": "precision_gripper",
            },
            {"robot": "robot_3", "action": "MoveTo", "target": "right_center"},
            {"robot": "robot_3", "action": "Pick", "object": "shaft_1"},
            {"robot": "robot_3", "action": "MoveTo", "target": "bearing_block"},
            {
                "robot": "robot_3",
                "action": "Insert",
                "object": "shaft_1",
                "target": "bearing_block",
                "tool": "precision_gripper",
            },
        ],
        "success_conditions": [
            {"object": "gear_1", "target": "chassis"},
            {"object": "shaft_1", "target": "bearing_block"},
        ],
    }


def four_robot_assembly_payload() -> Dict[str, Any]:
    return {
        "task_summary": "Two feeders and two specialists cooperate on a dual-part assembly.",
        "robots": [
            {
                "name": "robot_1",
                "role": "gear_feeder",
                "start_location": "left_mid",
                "can_move": True,
            },
            {
                "name": "robot_2",
                "role": "shaft_feeder",
                "start_location": "lower_mid",
                "can_move": True,
            },
            {
                "name": "robot_3",
                "role": "gear_assembler",
                "start_location": "center",
                "can_move": True,
            },
            {
                "name": "robot_4",
                "role": "shaft_assembler",
                "start_location": "upper_mid",
                "can_move": True,
            },
        ],
        "objects": [
            {"name": "gear_1", "kind": "gear", "shape": "circle"},
            {"name": "shaft_1", "kind": "shaft", "shape": "square"},
        ],
        "plan": [
            {"robot": "robot_1", "action": "Pick", "object": "gear_1"},
            {"robot": "robot_1", "action": "MoveTo", "target": "center"},
            {
                "robot": "robot_1",
                "action": "Handoff",
                "object": "gear_1",
                "recipient": "robot_3",
                "location": "center",
            },
            {"robot": "robot_2", "action": "Pick", "object": "shaft_1"},
            {"robot": "robot_2", "action": "MoveTo", "target": "center"},
            {
                "robot": "robot_2",
                "action": "Handoff",
                "object": "shaft_1",
                "recipient": "robot_4",
                "location": "center",
            },
            {"robot": "robot_3", "action": "MoveTo", "target": "tool_rack"},
            {"robot": "robot_3", "action": "ChangeTool", "tool": "precision_gripper"},
            {"robot": "robot_3", "action": "MoveTo", "target": "chassis"},
            {
                "robot": "robot_3",
                "action": "Insert",
                "object": "gear_1",
                "target": "chassis",
                "tool": "precision_gripper",
            },
            {"robot": "robot_4", "action": "MoveTo", "target": "tool_rack"},
            {"robot": "robot_4", "action": "ChangeTool", "tool": "precision_gripper"},
            {"robot": "robot_4", "action": "MoveTo", "target": "bearing_block"},
            {
                "robot": "robot_4",
                "action": "Insert",
                "object": "shaft_1",
                "target": "bearing_block",
                "tool": "precision_gripper",
            },
        ],
        "success_conditions": [
            {"object": "gear_1", "target": "chassis"},
            {"object": "shaft_1", "target": "bearing_block"},
        ],
    }


GRIDWORLD_PRESETS: List[GridWorldPreset] = [
    GridWorldPreset(
        name="Custom",
        scenario_text=CUSTOM_GRIDWORLD_SCENARIO_PLACEHOLDER,
        num_robots=3,
        num_circles=3,
        layout_name="open_room",
    ),
    GridWorldPreset(
        name="Distributed Corners",
        scenario_text="All robots collect circles and place them in different corners of the room.",
        num_robots=3,
        num_circles=3,
        layout_name="open_room",
    ),
    GridWorldPreset(
        name="Two Move One Waits",
        scenario_text="Two robots collect circles and place them at the bottom corners, while the third robot just waits at the center.",
        num_robots=3,
        num_circles=2,
        layout_name="open_room",
    ),
    GridWorldPreset(
        name="Stationary Receiver",
        scenario_text="Two robots sequentially collect circles and give them one at a time to the third robot, which waits at the center and places each circle there before receiving the next.",
        num_robots=3,
        num_circles=2,
        layout_name="handoff_hall",
    ),
    GridWorldPreset(
        name="Three Robot Relay",
        scenario_text="First robot hands a circle to the second robot at the center, then the second robot carries it to the top-left corner and gives it to the third robot, who places it at the bottom-left corner.",
        num_robots=3,
        num_circles=1,
        layout_name="four_rooms",
    ),
    GridWorldPreset(
        name="Parallel Rooms",
        scenario_text="All robots collect circles and distribute them across different rooms while avoiding redundant handoffs.",
        num_robots=4,
        num_circles=4,
        layout_name="four_rooms",
    ),
    GridWorldPreset(
        name="Wall Sweep",
        scenario_text="All robots collect circles and place them along the sides.",
        num_robots=8,
        num_circles=8,
        layout_name="four_rooms",
    ),
    GridWorldPreset(
        name="Precision Insert",
        scenario_text="Robot 1 moves to the tool rack, changes to the precision gripper, picks up the gear, and inserts it into the chassis.",
        num_robots=1,
        num_circles=1,
        layout_name="open_room",
    ),
    GridWorldPreset(
        name="Relay Insert",
        scenario_text="Robot 1 picks up the shaft, hands it to robot 2 at the center, then robot 2 changes to the precision gripper and inserts the shaft into the bearing block.",
        num_robots=2,
        num_circles=1,
        layout_name="handoff_hall",
    ),
    GridWorldPreset(
        name="Three Robot Assembly",
        scenario_text="Three robots assemble two parts: robot 1 and robot 2 fetch their assigned parts in parallel, and robot 3 prepares its tool and inserts both parts into the correct fixtures.",
        num_robots=3,
        num_circles=2,
        layout_name="open_room",
    ),
    GridWorldPreset(
        name="Four Robot Assembly",
        scenario_text="Four robots assemble two parts: robot 1 feeds a gear to robot 3, robot 2 feeds a shaft to robot 4, and robots 3 and 4 insert both parts into the correct fixtures.",
        num_robots=4,
        num_circles=2,
        layout_name="open_room",
    ),
]


def resolve_preset_payload(
    preset_name: str,
    scenario_text: str,
    num_robots: int,
    num_objects: int,
    layout_name: str,
) -> Optional[Dict[str, Any]]:
    preset = next((item for item in GRIDWORLD_PRESETS if item.name == preset_name), None)
    if preset is None:
        return None

    if (
        scenario_text.strip() != preset.scenario_text.strip()
        or num_robots != preset.num_robots
        or num_objects != preset.num_circles
        or layout_name != preset.layout_name
    ):
        return None

    if preset_name == "Precision Insert":
        return precision_insert_payload()
    if preset_name == "Relay Insert":
        return relay_insert_payload()
    if preset_name == "Three Robot Assembly":
        return three_robot_assembly_payload()
    if preset_name == "Four Robot Assembly":
        return four_robot_assembly_payload()
    return None
