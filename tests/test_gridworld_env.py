from collections import deque
import unittest

from src.gridworld_env import (
    DEFAULT_TYPED_SCENARIO,
    build_env_from_typed_scenario,
    named_layout_locations,
    resolve_typed_scenario_text,
)
from src.gridworld_layouts import GridLayout, layout_registry
from src.gridworld_presets import (
    four_robot_assembly_payload,
    relay_insert_payload,
    three_robot_assembly_payload,
)


def distributed_corners_payload():
    return {
        "task_summary": "Two collectors move circles while the third robot waits.",
        "robots": [
            {
                "name": "robot_1",
                "role": "collector",
                "start_location": "left_mid",
                "can_move": True,
            },
            {
                "name": "robot_2",
                "role": "collector",
                "start_location": "lower_mid",
                "can_move": True,
            },
            {
                "name": "robot_3",
                "role": "waiting_observer",
                "start_location": "center",
                "can_move": False,
            },
        ],
        "objects": [
            {"name": "circle_1", "shape": "circle"},
            {"name": "circle_2", "shape": "circle"},
        ],
        "plan": [
            {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
            {"robot": "robot_1", "action": "MoveTo", "target": "top_right"},
            {"robot": "robot_1", "action": "Place", "object": "circle_1", "target": "top_right"},
            {"robot": "robot_2", "action": "Pick", "object": "circle_2"},
            {"robot": "robot_2", "action": "MoveTo", "target": "bottom_right"},
            {"robot": "robot_2", "action": "Place", "object": "circle_2", "target": "bottom_right"},
        ],
        "success_conditions": [
            {"object": "circle_1", "target": "top_right"},
            {"object": "circle_2", "target": "bottom_right"},
        ],
    }


def waiting_receiver_payload():
    return {
        "task_summary": "Two collectors hand circles to a stationary center receiver.",
        "robots": [
            {
                "name": "robot_1",
                "role": "collector",
                "start_location": "left_mid",
                "can_move": True,
            },
            {
                "name": "robot_2",
                "role": "collector",
                "start_location": "lower_mid",
                "can_move": True,
            },
            {
                "name": "robot_3",
                "role": "waiting_receiver",
                "start_location": "center",
                "can_move": False,
            },
        ],
        "objects": [
            {"name": "circle_1", "shape": "circle"},
            {"name": "circle_2", "shape": "circle"},
        ],
        "plan": [
            {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
            {"robot": "robot_1", "action": "MoveTo", "target": "center"},
            {
                "robot": "robot_1",
                "action": "Handoff",
                "object": "circle_1",
                "recipient": "robot_3",
                "location": "center",
            },
            {"robot": "robot_3", "action": "Place", "object": "circle_1", "target": "center"},
            {"robot": "robot_2", "action": "Pick", "object": "circle_2"},
            {"robot": "robot_2", "action": "MoveTo", "target": "center"},
            {
                "robot": "robot_2",
                "action": "Handoff",
                "object": "circle_2",
                "recipient": "robot_3",
                "location": "center",
            },
            {"robot": "robot_3", "action": "Place", "object": "circle_2", "target": "center"},
        ],
        "success_conditions": [
            {"object": "circle_1", "target": "center"},
            {"object": "circle_2", "target": "center"},
        ],
    }


def precision_insert_payload():
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


class FakePlanner:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def plan_gridworld_task(self, instruction, num_robots, layout_name, available_locations, num_circles=None):
        self.calls.append(
            {
                "instruction": instruction,
                "num_robots": num_robots,
                "num_circles": num_circles,
                "layout_name": layout_name,
                "available_locations": list(available_locations),
            }
        )
        return self.payload


class GridWorldEnvTests(unittest.TestCase):
    def _layout_has_path(self, layout: GridLayout, start, goal) -> bool:
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            if current == goal:
                return True

            x, y = current
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nx, ny = neighbor
                if not (0 <= nx < layout.width and 0 <= ny < layout.height):
                    continue
                if neighbor in visited or neighbor in layout.walls:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return False

    def _assert_layout_points_connected(self, layout_name: str) -> None:
        layout = layout_registry()[layout_name]
        interesting_points = list(layout.preferred_robot_spawns) + list(layout.preferred_object_spawns)
        interesting_points.extend(named_layout_locations(layout).values())

        open_points = []
        for point in interesting_points:
            if point not in layout.walls and point not in open_points:
                open_points.append(point)

        self.assertTrue(open_points, "Expected at least one open point in layout '{}'.".format(layout_name))

        anchor = open_points[0]
        for point in open_points[1:]:
            self.assertTrue(
                self._layout_has_path(layout, anchor, point),
                "Expected '{}' to connect {} to {}.".format(layout_name, anchor, point),
            )

    def test_empty_scenario_text_uses_default(self):
        self.assertEqual(resolve_typed_scenario_text(""), DEFAULT_TYPED_SCENARIO)
        self.assertEqual(resolve_typed_scenario_text(None), DEFAULT_TYPED_SCENARIO)

    def test_build_env_calls_llm_planner(self):
        fake_planner = FakePlanner(distributed_corners_payload())

        env = build_env_from_typed_scenario(
            "Only two robots move while the third waits.",
            num_robots=3,
            num_circles=2,
            layout_name="open_room",
            seed=0,
            planner=fake_planner,
        )

        self.assertEqual(len(fake_planner.calls), 1)
        self.assertEqual(fake_planner.calls[0]["num_robots"], 3)
        self.assertEqual(fake_planner.calls[0]["num_circles"], 2)
        self.assertEqual(env.scenario.task_summary, distributed_corners_payload()["task_summary"])
        self.assertEqual(len(env.build_symbolic_plan()), 6)

    def test_environment_supports_idle_robot_and_different_targets(self):
        env = build_env_from_typed_scenario(
            "Only two robots move while the third waits.",
            num_robots=3,
            layout_name="open_room",
            seed=0,
            scenario_payload=distributed_corners_payload(),
        )

        waiting_start = env.resolve_named_position("center")
        result = env.run(max_steps=50)

        self.assertTrue(result.completed)
        self.assertTrue(env.all_objects_delivered())
        self.assertEqual(env.robots["robot_3"].position, waiting_start)
        self.assertEqual(env.objects["circle_1"].position, env.resolve_named_position("top_right"))
        self.assertEqual(env.objects["circle_2"].position, env.resolve_named_position("bottom_right"))

    def test_environment_supports_stationary_receiver_handoff(self):
        env = build_env_from_typed_scenario(
            "Two robots collect circles and give them to the third one which just waits.",
            num_robots=3,
            layout_name="open_room",
            seed=0,
            scenario_payload=waiting_receiver_payload(),
        )

        receiver_start = env.resolve_named_position("center")
        result = env.run(max_steps=60)

        self.assertTrue(result.completed)
        self.assertTrue(env.all_objects_delivered())
        self.assertEqual(env.robots["robot_3"].position, receiver_start)
        self.assertEqual(env.objects["circle_1"].position, receiver_start)
        self.assertEqual(env.objects["circle_2"].position, receiver_start)

    def test_environment_supports_precision_insert_scenario(self):
        env = build_env_from_typed_scenario(
            "Robot 1 changes tools and inserts a gear into the chassis.",
            num_robots=1,
            layout_name="open_room",
            seed=0,
            scenario_payload=precision_insert_payload(),
        )

        tree = env.build_behavior_tree()
        tree.setup()

        result = env.run(max_steps=60)

        self.assertTrue(result.completed)
        self.assertTrue(env.all_objects_delivered())
        self.assertEqual(env.robots["robot_1"].current_tool, "precision_gripper")
        self.assertEqual(env.objects["gear_1"].inserted_target, "chassis")

        frame = result.history[0].frame
        fixed_location_names = {location.name for location in frame.fixed_locations}
        self.assertIn("tool_rack", fixed_location_names)
        self.assertIn("chassis", fixed_location_names)

    def test_environment_supports_relay_insert_scenario(self):
        env = build_env_from_typed_scenario(
            "Robot 1 picks up the shaft, hands it to robot 2 at the center, then robot 2 changes to the precision gripper and inserts the shaft into the bearing block.",
            num_robots=2,
            layout_name="handoff_hall",
            seed=0,
            scenario_payload=relay_insert_payload(),
        )

        result = env.run(max_steps=80)

        self.assertTrue(result.completed)
        self.assertTrue(env.all_objects_delivered())
        self.assertEqual(env.robots["robot_2"].current_tool, "precision_gripper")
        self.assertEqual(env.objects["shaft_1"].inserted_target, "bearing_block")

    def test_environment_supports_three_robot_assembly_scenario(self):
        env = build_env_from_typed_scenario(
            "Three robots assemble two parts with two feeders and one assembler.",
            num_robots=3,
            layout_name="open_room",
            seed=0,
            scenario_payload=three_robot_assembly_payload(),
        )

        tree = env.build_behavior_tree()
        phase_terminals = [
            {segment.terminal_action for segment in phase}
            for phase in tree.phase_segments
        ]

        result = env.run(max_steps=120)

        self.assertTrue(result.completed)
        self.assertTrue(env.all_objects_delivered())
        self.assertEqual(env.robots["robot_3"].current_tool, "precision_gripper")
        self.assertEqual(env.objects["gear_1"].inserted_target, "chassis")
        self.assertEqual(env.objects["shaft_1"].inserted_target, "bearing_block")
        self.assertEqual(
            {step["robot"] for step in env.build_symbolic_plan()},
            {"robot_1", "robot_2", "robot_3"},
        )
        self.assertIn({"Place", "ChangeTool"}, phase_terminals)

    def test_environment_supports_four_robot_assembly_scenario(self):
        env = build_env_from_typed_scenario(
            "Four robots assemble two parts with two feeders and two assemblers.",
            num_robots=4,
            layout_name="open_room",
            seed=0,
            scenario_payload=four_robot_assembly_payload(),
        )

        result = env.run(max_steps=120)

        self.assertTrue(result.completed)
        self.assertTrue(env.all_objects_delivered())
        self.assertEqual(env.robots["robot_3"].current_tool, "precision_gripper")
        self.assertEqual(env.robots["robot_4"].current_tool, "precision_gripper")
        self.assertEqual(env.objects["gear_1"].inserted_target, "chassis")
        self.assertEqual(env.objects["shaft_1"].inserted_target, "bearing_block")
        self.assertEqual(
            {step["robot"] for step in env.build_symbolic_plan()},
            {"robot_1", "robot_2", "robot_3", "robot_4"},
        )

    def test_environment_allows_shared_start_location(self):
        env = build_env_from_typed_scenario(
            "Two robots can start together at the center.",
            num_robots=3,
            layout_name="open_room",
            seed=0,
            scenario_payload={
                "task_summary": "Two robots share the center start while one heads out to place a circle.",
                "robots": [
                    {"name": "robot_1", "role": "receiver", "start_location": "center", "can_move": False},
                    {"name": "robot_2", "role": "collector", "start_location": "center", "can_move": True},
                    {"name": "robot_3", "role": "observer", "start_location": "left_mid", "can_move": False},
                ],
                "objects": [
                    {"name": "circle_1", "shape": "circle"},
                ],
                "plan": [
                    {"robot": "robot_2", "action": "Pick", "object": "circle_1"},
                    {"robot": "robot_2", "action": "MoveTo", "target": "top_right"},
                    {"robot": "robot_2", "action": "Place", "object": "circle_1", "target": "top_right"},
                ],
                "success_conditions": [
                    {"object": "circle_1", "target": "top_right"},
                ],
            },
        )

        center_position = env.resolve_named_position("center")
        self.assertEqual(env.robots["robot_1"].position, center_position)
        self.assertEqual(env.robots["robot_2"].position, center_position)
        self.assertIn("*", env.render())

        result = env.run(max_steps=40)

        self.assertTrue(result.completed)
        self.assertEqual(env.robots["robot_1"].position, center_position)
        self.assertEqual(env.objects["circle_1"].position, env.resolve_named_position("top_right"))

    def test_split_room_layout_is_supported_with_llm_payload(self):
        env = build_env_from_typed_scenario(
            "All robots collect circles and place them in different corners.",
            num_robots=3,
            layout_name="split_room",
            seed=1,
            scenario_payload={
                "task_summary": "Robots distribute circles across corners in split_room.",
                "robots": [
                    {"name": "robot_1", "role": "collector", "start_location": "left_mid", "can_move": True},
                    {"name": "robot_2", "role": "collector", "start_location": "upper_mid", "can_move": True},
                    {"name": "robot_3", "role": "collector", "start_location": "lower_mid", "can_move": True},
                ],
                "objects": [
                    {"name": "circle_1", "shape": "circle"},
                    {"name": "circle_2", "shape": "circle"},
                    {"name": "circle_3", "shape": "circle"},
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
                    {"robot": "robot_3", "action": "Place", "object": "circle_3", "target": "bottom_right"},
                ],
                "success_conditions": [
                    {"object": "circle_1", "target": "top_left"},
                    {"object": "circle_2", "target": "top_right"},
                    {"object": "circle_3", "target": "bottom_right"},
                ],
            },
        )

        self.assertEqual(env.layout.name, "split_room")
        self.assertIn("#", env.render())
        self.assertEqual(len(env.build_symbolic_plan()), 9)

        result = env.run(max_steps=90)

        self.assertTrue(result.completed)
        self.assertEqual(env.objects["circle_1"].position, env.resolve_named_position("top_left"))
        self.assertEqual(env.objects["circle_2"].position, env.resolve_named_position("top_right"))
        self.assertEqual(env.objects["circle_3"].position, env.resolve_named_position("bottom_right"))

    def test_handoff_hall_layout_supports_stationary_receiver_handoff(self):
        env = build_env_from_typed_scenario(
            "Two robots sequentially give circles to a stationary center receiver.",
            num_robots=3,
            layout_name="handoff_hall",
            seed=0,
            scenario_payload=waiting_receiver_payload(),
        )

        result = env.run(max_steps=90)
        receiver_start = env.resolve_named_position("center")

        self.assertTrue(result.completed)
        self.assertEqual(env.robots["robot_3"].position, receiver_start)
        self.assertEqual(env.objects["circle_1"].position, receiver_start)
        self.assertEqual(env.objects["circle_2"].position, receiver_start)

    def test_split_room_layout_points_are_connected(self):
        self._assert_layout_points_connected("split_room")

    def test_handoff_hall_layout_points_are_connected(self):
        self._assert_layout_points_connected("handoff_hall")

    def test_four_rooms_layout_produces_visual_frame_data(self):
        env = build_env_from_typed_scenario(
            "Robots relay a circle across rooms.",
            num_robots=3,
            layout_name="four_rooms",
            seed=2,
            scenario_payload={
                "task_summary": "A chained relay moves one circle into another room.",
                "robots": [
                    {"name": "robot_1", "role": "giver", "start_location": "left_mid", "can_move": True},
                    {"name": "robot_2", "role": "relay", "start_location": "center", "can_move": True},
                    {"name": "robot_3", "role": "receiver", "start_location": "upper_mid", "can_move": True},
                ],
                "objects": [
                    {"name": "circle_1", "shape": "circle"},
                ],
                "plan": [
                    {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
                    {"robot": "robot_1", "action": "MoveTo", "target": "center"},
                    {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_2", "location": "center"},
                    {"robot": "robot_2", "action": "MoveTo", "target": "top_right"},
                    {"robot": "robot_2", "action": "Handoff", "object": "circle_1", "recipient": "robot_3", "location": "top_right"},
                    {"robot": "robot_3", "action": "MoveTo", "target": "bottom_left"},
                    {"robot": "robot_3", "action": "Place", "object": "circle_1", "target": "bottom_left"},
                ],
                "success_conditions": [
                    {"object": "circle_1", "target": "bottom_left"},
                ],
            },
        )

        result = env.run(max_steps=80)

        self.assertEqual(env.layout.name, "four_rooms")
        self.assertTrue(result.history)
        self.assertGreaterEqual(result.history[0].frame.total_phases, 1)
        self.assertEqual(result.history[0].frame.width, env.layout.width)
        self.assertTrue(result.completed)


if __name__ == "__main__":
    unittest.main()
