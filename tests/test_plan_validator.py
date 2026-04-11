import unittest

from src.plan_validator import validate_reactive_plan


class PlanValidatorTests(unittest.TestCase):
    def test_warns_on_conflicting_move_and_terminal_targets(self):
        plan = [
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "chassis"},
            {"action": "Insert", "object": "gear", "target": "chassis2"},
        ]

        warnings = validate_reactive_plan(plan)

        self.assertEqual(len(warnings), 1)
        self.assertIn("MoveTo steps target step 2 -> chassis", warnings[0])
        self.assertIn("Insert(gear, chassis2)", warnings[0])

    def test_accepts_consistent_transfer_segment(self):
        plan = [
            {"action": "Pick", "object": "gear"},
            {"action": "MoveTo", "target": "chassis"},
            {"action": "Insert", "object": "gear", "target": "chassis"},
        ]

        warnings = validate_reactive_plan(plan)

        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
