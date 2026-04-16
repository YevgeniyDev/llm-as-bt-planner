import contextlib
import io
import unittest

import py_trees

from src.multi_robot_actions import RobotProfile
from src.multi_robot_planner import build_multi_robot_tree_from_json


class MultiRobotBTTests(unittest.TestCase):
    def run_quietly(self, callback):
        with contextlib.redirect_stdout(io.StringIO()):
            return callback()

    def tick_tree(self, tree, ticks):
        self.run_quietly(tree.setup)
        for _ in range(ticks):
            self.run_quietly(tree.tick)

    def tick_until_terminal(self, tree, max_ticks=20):
        self.run_quietly(tree.setup)
        for _ in range(max_ticks):
            self.run_quietly(tree.tick)
            if tree.root.status in {
                py_trees.common.Status.SUCCESS,
                py_trees.common.Status.FAILURE,
            }:
                return tree.root.status
        self.fail("Tree did not finish within {} ticks.".format(max_ticks))

    def test_distinct_object_segments_run_in_parallel_phase(self):
        plan = [
            {"action": "Pick", "object": "screwdriver"},
            {"action": "MoveTo", "target": "panel"},
            {"action": "Place", "object": "screwdriver", "target": "panel"},
            {"action": "Pick", "object": "hammer"},
            {"action": "MoveTo", "target": "table"},
            {"action": "Place", "object": "hammer", "target": "table"},
        ]
        profiles = [
            RobotProfile(name="robot1", priority=0),
            RobotProfile(name="robot2", priority=1),
        ]

        tree = build_multi_robot_tree_from_json(plan, robot_profiles=profiles)
        phase = tree.root.children[0]

        self.assertIsInstance(tree.root, py_trees.composites.Sequence)
        self.assertIsInstance(phase, py_trees.composites.Parallel)
        self.assertEqual(len(phase.children), 2)

        self.run_quietly(tree.setup)
        self.run_quietly(tree.tick)

        world_state = tree.world_state
        self.assertEqual(sorted(world_state.intentions.keys()), ["robot1", "robot2"])

        for _ in range(5):
            self.run_quietly(tree.tick)

        self.assertEqual(tree.root.status, py_trees.common.Status.SUCCESS)
        self.assertTrue(world_state.is_object_at("screwdriver", "panel"))
        self.assertTrue(world_state.is_object_at("hammer", "table"))

    def test_single_goal_backups_are_suppressed_by_intention_sharing(self):
        plan = [
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "table"},
            {"action": "Place", "object": "gear", "target": "table"},
        ]
        profiles = [
            RobotProfile(name="robot1", priority=0),
            RobotProfile(name="robot2", priority=1),
        ]

        tree = build_multi_robot_tree_from_json(plan, robot_profiles=profiles)
        phase = tree.root.children[0]

        self.assertIsInstance(phase, py_trees.composites.Parallel)
        self.assertEqual(len(phase.children), 2)

        self.run_quietly(tree.setup)
        self.run_quietly(tree.tick)

        world_state = tree.world_state
        self.assertEqual(sorted(world_state.intentions.keys()), ["robot1"])
        self.assertIsNone(world_state.held_objects["robot2"])

        for _ in range(5):
            self.run_quietly(tree.tick)

        self.assertEqual(tree.root.status, py_trees.common.Status.SUCCESS)
        self.assertTrue(world_state.is_object_at("gear", "table"))

    def test_capability_filtering_keeps_insert_goal_on_capable_robot(self):
        plan = [
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "chassis"},
            {"action": "Insert", "object": "gear", "target": "chassis"},
        ]
        profiles = [
            RobotProfile(name="robot1", priority=0),
            RobotProfile(name="robot2", capabilities=("Pick", "MoveTo", "Place"), priority=1),
        ]

        tree = build_multi_robot_tree_from_json(plan, robot_profiles=profiles)
        phase = tree.root.children[0]

        self.assertIsInstance(phase, py_trees.composites.Parallel)
        self.assertEqual(len(phase.children), 1)
        self.assertEqual(phase.children[0].name, "Robot robot1")

        self.run_quietly(tree.setup)
        self.run_quietly(tree.tick)

        world_state = tree.world_state
        self.assertEqual(sorted(world_state.intentions.keys()), ["robot1"])

        for _ in range(5):
            self.run_quietly(tree.tick)

        self.assertEqual(tree.root.status, py_trees.common.Status.SUCCESS)
        self.assertTrue(world_state.is_inserted("gear", "chassis"))

    def test_explicit_robot_assignment_is_respected(self):
        plan = [
            {"robot": "robot2", "action": "Pick", "object": "gear"},
            {"robot": "robot2", "action": "MoveTo", "target": "table"},
            {"robot": "robot2", "action": "Place", "object": "gear", "target": "table"},
        ]
        profiles = [
            RobotProfile(name="robot1", priority=0),
            RobotProfile(name="robot2", priority=1),
        ]

        tree = build_multi_robot_tree_from_json(plan, robot_profiles=profiles)
        phase = tree.root.children[0]

        self.assertIsInstance(phase, py_trees.composites.Parallel)
        self.assertEqual(len(phase.children), 1)
        self.assertEqual(phase.children[0].name, "Robot robot2")

        final_status = self.tick_until_terminal(tree)

        self.assertEqual(final_status, py_trees.common.Status.SUCCESS)
        self.assertTrue(tree.world_state.is_object_at("gear", "table"))

    def test_tool_specific_insert_is_assigned_to_robot_with_tool_access(self):
        plan = [
            {"action": "ChangeTool", "tool": "precision_gripper"},
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "chassis"},
            {"action": "Insert", "object": "gear", "target": "chassis", "tool": "precision_gripper"},
        ]
        profiles = [
            RobotProfile(
                name="robot1",
                available_tools=("default_gripper",),
                priority=0,
            ),
            RobotProfile(
                name="robot2",
                available_tools=("default_gripper", "precision_gripper"),
                priority=1,
            ),
        ]

        tree = build_multi_robot_tree_from_json(plan, robot_profiles=profiles)
        phase = tree.root.children[0]

        self.assertIsInstance(phase, py_trees.composites.Parallel)
        self.assertEqual(len(phase.children), 1)
        self.assertEqual(phase.children[0].name, "Robot robot2")

        final_status = self.tick_until_terminal(tree)

        self.assertEqual(final_status, py_trees.common.Status.SUCCESS)
        self.assertTrue(tree.world_state.is_inserted("gear", "chassis"))
        self.assertEqual(tree.world_state.equipped_tools["robot2"], "precision_gripper")

    def test_handoff_to_specialist_with_tool_change_insert(self):
        plan = [
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "handoff_station"},
            {"action": "Handoff", "object": "gear", "recipient": "robot2", "location": "handoff_station"},
            {"action": "ChangeTool", "tool": "precision_gripper"},
            {"action": "MoveTo", "target": "chassis"},
            {"action": "Insert", "object": "gear", "target": "chassis", "tool": "precision_gripper"},
        ]
        profiles = [
            RobotProfile(
                name="robot1",
                capabilities=("Pick", "MoveTo", "Handoff"),
                available_tools=("default_gripper",),
                priority=0,
            ),
            RobotProfile(
                name="robot2",
                capabilities=("MoveTo", "Insert", "ChangeTool"),
                available_tools=("default_gripper", "precision_gripper"),
                priority=1,
            ),
        ]

        tree = build_multi_robot_tree_from_json(plan, robot_profiles=profiles)

        self.assertEqual(len(tree.root.children), 2)
        self.assertIsInstance(tree.root.children[0], py_trees.composites.Parallel)
        self.assertIsInstance(tree.root.children[1], py_trees.composites.Parallel)

        self.run_quietly(tree.setup)
        self.run_quietly(tree.tick)

        self.assertEqual(sorted(tree.world_state.intentions.keys()), ["robot2"])

        for _ in range(19):
            self.run_quietly(tree.tick)
            if tree.root.status in {
                py_trees.common.Status.SUCCESS,
                py_trees.common.Status.FAILURE,
            }:
                break

        self.assertEqual(tree.root.status, py_trees.common.Status.SUCCESS)
        self.assertTrue(tree.world_state.is_inserted("gear", "chassis"))
        self.assertEqual(tree.world_state.equipped_tools["robot2"], "precision_gripper")
        self.assertTrue(tree.world_state.is_hand_empty("robot1"))
        self.assertTrue(tree.world_state.is_hand_empty("robot2"))


if __name__ == "__main__":
    unittest.main()
