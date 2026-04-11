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


if __name__ == "__main__":
    unittest.main()
