import unittest

from src.llm_client import LLMTaskPlanner


class LLMClientCanonicalizationTests(unittest.TestCase):
    def test_recursive_style_action_strings_are_canonicalized(self):
        planner = object.__new__(LLMTaskPlanner)

        plan = planner._canonicalize_plan(
            [
                "Pick(gear)",
                "MoveTo(chassis)",
                "Insert(gear, chassis)",
            ]
        )

        self.assertEqual(
            plan,
            [
                {"action": "Pick", "object": "gear"},
                {"action": "MoveTo", "target": "chassis"},
                {"action": "Insert", "object": "gear", "target": "chassis"},
            ],
        )

    def test_tool_change_and_handoff_are_canonicalized(self):
        planner = object.__new__(LLMTaskPlanner)

        plan = planner._canonicalize_plan(
            [
                "ChangeTool(precision_gripper)",
                "Handoff(gear, robot2, handoff_station)",
            ]
        )

        self.assertEqual(
            plan,
            [
                {"action": "ChangeTool", "tool": "precision_gripper"},
                {
                    "action": "Handoff",
                    "object": "gear",
                    "recipient": "robot2",
                    "location": "handoff_station",
                },
            ],
        )

    def test_handoff_requires_a_location(self):
        planner = object.__new__(LLMTaskPlanner)

        with self.assertRaisesRegex(ValueError, "location"):
            planner._validate_plan(
                [
                    {
                        "action": "Handoff",
                        "object": "gear",
                        "recipient": "robot2",
                    }
                ]
            )

    def test_gridworld_spec_keeps_robot_assignments(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "Robot 1 works while robot 2 waits.",
            "robots": [
                {"name": "robot_1", "role": "collector", "start_location": "left_mid", "can_move": True},
                {"name": "robot_2", "role": "waiting_observer", "start_location": "center", "can_move": False},
            ],
            "objects": [
                {"name": "circle_1", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "Pick(circle_1)"},
                {"robot": "robot_1", "action": "MoveTo(top_right)"},
                {"robot": "robot_1", "action": "Place(circle_1, top_right)"},
            ],
            "success_conditions": [
                {"object": "circle_1", "target": "top_right"},
            ],
        }

        spec = planner._parse_gridworld_spec(
            payload=payload,
            instruction="Robot 1 moves a circle while robot 2 waits.",
            num_robots=2,
            available_locations=["center", "left_mid", "top_right"],
        )

        self.assertEqual(spec["plan"][0]["robot"], "robot_1")
        self.assertEqual(spec["plan"][1]["target"], "top_right")
        self.assertFalse(spec["robots"][1]["can_move"])

    def test_gridworld_spec_repairs_missing_robot_entry_and_name_variants(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "A fourth robot relays the circle after a handoff.",
            "robots": [
                {"name": "Robot 1", "role": "collector", "start_location": "left_mid", "can_move": True},
                {"name": "robot_2", "role": "observer", "start_location": "upper_mid", "can_move": True},
                {"name": "robot_3", "role": "observer", "start_location": "lower_mid", "can_move": True},
            ],
            "objects": [
                {"name": "circle_1", "shape": "circle"},
            ],
            "plan": [
                {"robot": "Robot 1", "action": "Pick(circle_1)"},
                {"robot": "Robot 1", "action": "MoveTo(center)"},
                {"robot": "Robot 1", "action": "Handoff(circle_1, robot4, center)"},
                {"robot": "robot4", "action": "MoveTo(top_right)"},
                {"robot": "robot4", "action": "Place(circle_1, top_right)"},
            ],
            "success_conditions": [
                {"object": "circle_1", "target": "top_right"},
            ],
        }

        spec = planner._parse_gridworld_spec(
            payload=payload,
            instruction="Robot 1 hands a circle to robot 4, which places it at the top right.",
            num_robots=4,
            available_locations=["center", "left_mid", "upper_mid", "lower_mid", "top_right"],
        )

        self.assertEqual(
            [robot["name"] for robot in spec["robots"]],
            ["robot_1", "robot_2", "robot_3", "robot_4"],
        )
        self.assertEqual(spec["plan"][0]["robot"], "robot_1")
        self.assertEqual(spec["plan"][2]["recipient"], "robot_4")
        self.assertEqual(spec["plan"][3]["robot"], "robot_4")
        self.assertEqual(spec["robots"][-1]["start_location"], "center")

    def test_gridworld_spec_rejects_move_for_stationary_robot(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "Stationary robot should not move.",
            "robots": [
                {"name": "robot_1", "role": "waiting_receiver", "start_location": "center", "can_move": False},
            ],
            "objects": [
                {"name": "circle_1", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "MoveTo", "target": "top_right"},
            ],
            "success_conditions": [
                {"object": "circle_1", "target": "center"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "can_move=false"):
            planner._parse_gridworld_spec(
                payload=payload,
                instruction="Robot 1 waits.",
                num_robots=1,
                available_locations=["center", "top_right"],
            )

    def test_gridworld_spec_rejects_wrong_requested_object_count(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "Robot 1 moves one circle.",
            "robots": [
                {"name": "robot_1", "role": "collector", "start_location": "left_mid", "can_move": True},
            ],
            "objects": [
                {"name": "circle_1", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "Pick(circle_1)"},
                {"robot": "robot_1", "action": "MoveTo(top_right)"},
                {"robot": "robot_1", "action": "Place(circle_1, top_right)"},
            ],
            "success_conditions": [
                {"object": "circle_1", "target": "top_right"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "exactly 2 objects"):
            planner._parse_gridworld_spec(
                payload=payload,
                instruction="Move two circles.",
                num_robots=1,
                available_locations=["left_mid", "top_right"],
                num_circles=2,
            )

    def test_gridworld_spec_accepts_assembly_object_kind_and_insert(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "One robot inserts a gear into the chassis.",
            "robots": [
                {"name": "robot_1", "role": "assembler", "start_location": "left_mid", "can_move": True},
            ],
            "objects": [
                {"name": "gear_1", "kind": "gear", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "MoveTo", "target": "tool_rack"},
                {"robot": "robot_1", "action": "ChangeTool", "tool": "precision_gripper"},
                {"robot": "robot_1", "action": "Pick", "object": "gear_1"},
                {"robot": "robot_1", "action": "MoveTo", "target": "chassis"},
                {"robot": "robot_1", "action": "Insert", "object": "gear_1", "target": "chassis", "tool": "precision_gripper"},
            ],
            "success_conditions": [
                {"object": "gear_1", "target": "chassis"},
            ],
        }

        spec = planner._parse_gridworld_spec(
            payload=payload,
            instruction="Robot 1 inserts a gear into the chassis.",
            num_robots=1,
            available_locations=["left_mid", "tool_rack", "chassis"],
        )

        self.assertEqual(spec["objects"][0]["kind"], "gear")
        self.assertEqual(spec["plan"][-1]["action"], "Insert")

    def test_gridworld_spec_rejects_incompatible_insert_target(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "One robot tries to insert a gear into the wrong fixture.",
            "robots": [
                {"name": "robot_1", "role": "assembler", "start_location": "left_mid", "can_move": True},
            ],
            "objects": [
                {"name": "gear_1", "kind": "gear", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "Pick", "object": "gear_1"},
                {"robot": "robot_1", "action": "MoveTo", "target": "bearing_block"},
                {"robot": "robot_1", "action": "Insert", "object": "gear_1", "target": "bearing_block"},
            ],
            "success_conditions": [
                {"object": "gear_1", "target": "bearing_block"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "cannot be inserted into 'bearing_block'"):
            planner._parse_gridworld_spec(
                payload=payload,
                instruction="Insert the gear into the bearing block.",
                num_robots=1,
                available_locations=["left_mid", "bearing_block"],
            )

    def test_gridworld_handoff_normalization_drops_receiver_prep_move(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "Receiver move is redundant for handoff.",
            "robots": [
                {"name": "robot_1", "role": "giver", "start_location": "left_mid", "can_move": True},
                {"name": "robot_2", "role": "receiver", "start_location": "upper_mid", "can_move": True},
            ],
            "objects": [
                {"name": "circle_1", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
                {"robot": "robot_1", "action": "MoveTo", "target": "center"},
                {"robot": "robot_2", "action": "MoveTo", "target": "center"},
                {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_2", "location": "center"},
                {"robot": "robot_2", "action": "MoveTo", "target": "bottom_left"},
                {"robot": "robot_2", "action": "Place", "object": "circle_1", "target": "bottom_left"},
            ],
            "success_conditions": [
                {"object": "circle_1", "target": "bottom_left"},
            ],
        }

        normalized = planner._normalize_gridworld_payload_for_execution(payload)

        self.assertEqual(
            normalized["plan"],
            [
                {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
                {"robot": "robot_1", "action": "MoveTo", "target": "center"},
                {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_2", "location": "center"},
                {"robot": "robot_2", "action": "MoveTo", "target": "bottom_left"},
                {"robot": "robot_2", "action": "Place", "object": "circle_1", "target": "bottom_left"},
            ],
        )

    def test_gridworld_bt_compatibility_rejects_repeated_wrong_giver(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "Robot 1 cannot hand off the same circle twice.",
            "robots": [
                {"name": "robot_1", "role": "giver", "start_location": "left_mid", "can_move": True},
                {"name": "robot_2", "role": "relay", "start_location": "center", "can_move": True},
                {"name": "robot_3", "role": "receiver", "start_location": "top_right", "can_move": True},
            ],
            "objects": [
                {"name": "circle_1", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
                {"robot": "robot_1", "action": "MoveTo", "target": "center"},
                {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_2", "location": "center"},
                {"robot": "robot_1", "action": "MoveTo", "target": "top_right"},
                {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_3", "location": "top_right"},
            ],
            "success_conditions": [
                {"object": "circle_1", "target": "top_right"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "compatibility check failed"):
            planner._validate_gridworld_bt_compatibility(payload)

    def test_gridworld_normalization_serializes_stationary_receiver_handoffs(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "Two collectors hand circles to a waiting receiver.",
            "robots": [
                {"name": "robot_1", "role": "collector", "start_location": "left_mid", "can_move": True},
                {"name": "robot_2", "role": "collector", "start_location": "lower_mid", "can_move": True},
                {"name": "robot_3", "role": "waiting_receiver", "start_location": "center", "can_move": False},
            ],
            "objects": [
                {"name": "circle_1", "shape": "circle"},
                {"name": "circle_2", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
                {"robot": "robot_1", "action": "MoveTo", "target": "center"},
                {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_3", "location": "center"},
                {"robot": "robot_2", "action": "Pick", "object": "circle_2"},
                {"robot": "robot_2", "action": "MoveTo", "target": "center"},
                {"robot": "robot_2", "action": "Handoff", "object": "circle_2", "recipient": "robot_3", "location": "center"},
            ],
            "success_conditions": [
                {"object": "circle_1", "target": "center"},
                {"object": "circle_2", "target": "center"},
            ],
        }

        normalized = planner._normalize_gridworld_payload_for_execution(payload)

        self.assertEqual(
            normalized["plan"],
            [
                {"robot": "robot_1", "action": "Pick", "object": "circle_1"},
                {"robot": "robot_1", "action": "MoveTo", "target": "center"},
                {"robot": "robot_1", "action": "Handoff", "object": "circle_1", "recipient": "robot_3", "location": "center"},
                {"robot": "robot_3", "action": "Place", "object": "circle_1", "target": "center"},
                {"robot": "robot_2", "action": "Pick", "object": "circle_2"},
                {"robot": "robot_2", "action": "MoveTo", "target": "center"},
                {"robot": "robot_2", "action": "Handoff", "object": "circle_2", "recipient": "robot_3", "location": "center"},
                {"robot": "robot_3", "action": "Place", "object": "circle_2", "target": "center"},
            ],
        )

        planner._validate_gridworld_bt_compatibility(normalized)

    def test_gridworld_bt_compatibility_infers_tools_from_plan(self):
        planner = object.__new__(LLMTaskPlanner)

        payload = {
            "task_summary": "Tool-aware insert remains BT-compatible.",
            "robots": [
                {"name": "robot_1", "role": "assembler", "start_location": "left_mid", "can_move": True},
            ],
            "objects": [
                {"name": "gear_1", "kind": "gear", "shape": "circle"},
            ],
            "plan": [
                {"robot": "robot_1", "action": "MoveTo", "target": "tool_rack"},
                {"robot": "robot_1", "action": "ChangeTool", "tool": "precision_gripper"},
                {"robot": "robot_1", "action": "Pick", "object": "gear_1"},
                {"robot": "robot_1", "action": "MoveTo", "target": "chassis"},
                {"robot": "robot_1", "action": "Insert", "object": "gear_1", "target": "chassis", "tool": "precision_gripper"},
            ],
            "success_conditions": [
                {"object": "gear_1", "target": "chassis"},
            ],
        }

        planner._validate_gridworld_bt_compatibility(payload)


if __name__ == "__main__":
    unittest.main()
