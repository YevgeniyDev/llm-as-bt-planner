import contextlib
import io
import unittest
from unittest.mock import patch

import py_trees

from src.bt_builder import build_tree_from_json
from src.main import review_plan_with_human
from src.robot_actions import RobotWorldState


class StubPlanner:
    def __init__(self, revised_plan):
        self.revised_plan = revised_plan
        self.revision_calls = []

    def revise_plan(self, instruction, current_plan, human_feedback, tree_preview=None):
        self.revision_calls.append(
            {
                "instruction": instruction,
                "current_plan": current_plan,
                "human_feedback": human_feedback,
                "tree_preview": tree_preview,
            }
        )
        return self.revised_plan


class ReactiveBTTests(unittest.TestCase):
    def run_quietly(self, callback):
        with contextlib.redirect_stdout(io.StringIO()):
            return callback()

    def test_builder_emits_reactive_selectors_and_recovery_sequences(self):
        plan = [
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "table"},
            {"action": "Place", "object": "gear", "target": "table"},
        ]
        tree = build_tree_from_json(plan)

        self.assertIsInstance(tree.root, py_trees.composites.Sequence)
        self.assertFalse(tree.root.memory)
        self.assertIsInstance(tree.root.children[0], py_trees.composites.Selector)
        self.assertIsInstance(tree.root.children[1], py_trees.composites.Selector)
        self.assertIsInstance(tree.root.children[2], py_trees.composites.Selector)
        self.assertEqual(tree.root.children[2].children[0].name, "ObjectAt(gear, table)")
        self.assertEqual(tree.root.children[2].children[1].name, "RecoverThenPlace(gear)")
        self.assertIsInstance(tree.root.children[2].children[1], py_trees.composites.Sequence)

    def test_pick_executes_when_object_is_not_already_held(self):
        plan = [{"action": "Pick", "object": "gear"}]
        state = RobotWorldState.from_plan(plan)
        tree = build_tree_from_json(plan, world_state=state)

        self.run_quietly(tree.setup)
        self.run_quietly(tree.tick)

        pick_step = tree.root.children[0]
        self.assertEqual(tree.root.status, py_trees.common.Status.RUNNING)
        self.assertEqual(pick_step.children[0].status, py_trees.common.Status.FAILURE)
        self.assertEqual(pick_step.children[1].status, py_trees.common.Status.RUNNING)

        self.run_quietly(tree.tick)

        self.assertEqual(tree.root.status, py_trees.common.Status.SUCCESS)
        self.assertEqual(state.held_object, "gear")

    def test_pick_is_skipped_when_precondition_is_already_satisfied(self):
        plan = [{"action": "Pick", "object": "gear"}]
        state = RobotWorldState.from_plan(plan)
        state.held_object = "gear"
        state.object_locations.pop("gear", None)
        tree = build_tree_from_json(plan, world_state=state)

        self.run_quietly(tree.setup)
        self.run_quietly(tree.tick)

        pick_step = tree.root.children[0]
        self.assertEqual(tree.root.status, py_trees.common.Status.SUCCESS)
        self.assertEqual(pick_step.children[0].status, py_trees.common.Status.SUCCESS)
        self.assertEqual(pick_step.children[1].status, py_trees.common.Status.INVALID)

    def test_dynamic_failure_restarts_the_recovery_path(self):
        plan = [
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "table"},
            {"action": "Place", "object": "gear", "target": "table"},
        ]
        state = RobotWorldState.from_plan(plan)
        tree = build_tree_from_json(plan, world_state=state)

        self.run_quietly(tree.setup)
        self.run_quietly(tree.tick)
        self.run_quietly(tree.tick)
        self.run_quietly(tree.tick)

        self.assertEqual(state.held_object, "gear")
        self.assertEqual(state.robot_location, "table")

        dropped_object = state.drop_held_object("staging_area")
        self.assertEqual(dropped_object, "gear")
        self.assertIsNone(state.held_object)

        self.run_quietly(tree.tick)

        first_step = tree.root.children[0]
        self.assertEqual(tree.root.status, py_trees.common.Status.RUNNING)
        self.assertEqual(first_step.children[-1].status, py_trees.common.Status.RUNNING)

        self.run_quietly(tree.tick)
        self.run_quietly(tree.tick)

        self.assertEqual(tree.root.status, py_trees.common.Status.SUCCESS)
        self.assertTrue(state.is_object_at("gear", "table"))
        self.assertIsNone(state.held_object)

    def test_scheme3_review_uses_human_feedback_to_revise_plan(self):
        initial_plan = [{"action": "Pick", "object": "gear"}]
        revised_plan = [
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "chassis"},
            {"action": "Insert", "object": "gear", "target": "chassis"},
        ]
        planner = StubPlanner(revised_plan)

        with patch("builtins.input", side_effect=["n", "Use an insertion plan for the chassis.", "y"]):
            final_plan, tree = self.run_quietly(
                lambda: review_plan_with_human(
                    planner=planner,
                    instruction="Assemble the gearbox.",
                    plan=initial_plan,
                )
            )

        self.assertEqual(final_plan, revised_plan)
        self.assertEqual(len(planner.revision_calls), 1)
        self.assertIn("insertion plan", planner.revision_calls[0]["human_feedback"])
        self.assertIn("Step 3 Insert gear", tree.root.children[2].name)


if __name__ == "__main__":
    unittest.main()
