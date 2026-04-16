import unittest
from unittest import mock

from src.gridworld_app import GridWorldTesterApp, resolve_gridworld_app_scenario_text
from src.gridworld_env import DEFAULT_TYPED_SCENARIO
from src.gridworld_presets import (
    CUSTOM_GRIDWORLD_SCENARIO_PLACEHOLDER,
    GRIDWORLD_PRESETS,
    resolve_preset_payload,
)


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value


class GridWorldAppTests(unittest.TestCase):
    def test_custom_placeholder_falls_back_to_default_scenario(self):
        self.assertEqual(
            resolve_gridworld_app_scenario_text(CUSTOM_GRIDWORLD_SCENARIO_PLACEHOLDER),
            DEFAULT_TYPED_SCENARIO,
        )

    def test_custom_text_is_preserved_when_user_types_real_instruction(self):
        instruction = "All robots collect circles and place them in different corners."
        self.assertEqual(resolve_gridworld_app_scenario_text(instruction), instruction)

    def test_relay_insert_preset_uses_built_in_payload_when_unchanged(self):
        preset = next(item for item in GRIDWORLD_PRESETS if item.name == "Relay Insert")
        payload = resolve_preset_payload(
            preset_name=preset.name,
            scenario_text=preset.scenario_text,
            num_robots=preset.num_robots,
            num_objects=preset.num_circles,
            layout_name=preset.layout_name,
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["objects"][0]["name"], "shaft_1")
        self.assertEqual(payload["plan"][-1]["action"], "Insert")

    def test_four_robot_assembly_preset_uses_built_in_payload_when_unchanged(self):
        preset = next(item for item in GRIDWORLD_PRESETS if item.name == "Four Robot Assembly")
        payload = resolve_preset_payload(
            preset_name=preset.name,
            scenario_text=preset.scenario_text,
            num_robots=preset.num_robots,
            num_objects=preset.num_circles,
            layout_name=preset.layout_name,
        )

        self.assertIsNotNone(payload)
        self.assertEqual({step["robot"] for step in payload["plan"]}, {"robot_1", "robot_2", "robot_3", "robot_4"})
        self.assertEqual(len(payload["success_conditions"]), 2)

    def test_preset_payload_is_disabled_after_user_edits_instruction(self):
        preset = next(item for item in GRIDWORLD_PRESETS if item.name == "Relay Insert")
        payload = resolve_preset_payload(
            preset_name=preset.name,
            scenario_text=preset.scenario_text + " extra",
            num_robots=preset.num_robots,
            num_objects=preset.num_circles,
            layout_name=preset.layout_name,
        )

        self.assertIsNone(payload)

    def test_completion_popup_is_shown_once_for_success(self):
        app = object.__new__(GridWorldTesterApp)
        app.last_result = type(
            "Result",
            (),
            {"history": [object(), object()], "completed": True, "steps_run": 12},
        )()
        app.current_frame_index = 1
        app.completion_popup_shown = False

        with mock.patch("src.gridworld_app.messagebox.showinfo") as showinfo:
            app._show_completion_popup_if_needed()
            app._show_completion_popup_if_needed()

        showinfo.assert_called_once()
        self.assertTrue(app.completion_popup_shown)

    def test_completion_popup_is_warning_for_incomplete_run(self):
        app = object.__new__(GridWorldTesterApp)
        app.last_result = type(
            "Result",
            (),
            {"history": [object(), object(), object()], "completed": False, "steps_run": 70},
        )()
        app.current_frame_index = 2
        app.completion_popup_shown = False

        with mock.patch("src.gridworld_app.messagebox.showwarning") as showwarning:
            app._show_completion_popup_if_needed()

        showwarning.assert_called_once()
        self.assertTrue(app.completion_popup_shown)

    def test_clear_loaded_scenario_resets_previous_run_state(self):
        app = object.__new__(GridWorldTesterApp)
        app.last_result = object()
        app.current_env = object()
        app.current_frame_index = 8
        app.completion_popup_shown = True
        app.current_tree_preview = "tree"
        app.current_provider = "provider"
        app.current_model = "model"
        app.tick_var = FakeVar("Tick: 8/8")
        app.phase_var = FakeVar("Phase: 1/1")
        app.result_var = FakeVar("Result: SUCCESS")
        app.status_var = FakeVar("Loaded 9 frames using provider.")
        app._populate_summary_placeholder = mock.Mock()
        app._populate_plan_text = mock.Mock()
        app._populate_state_text = mock.Mock()
        app._update_event_panel = mock.Mock()
        app._draw_placeholder = mock.Mock()

        app._clear_loaded_scenario(status_message="Scenario failed to build.")

        self.assertIsNone(app.last_result)
        self.assertIsNone(app.current_env)
        self.assertEqual(app.current_frame_index, 0)
        self.assertFalse(app.completion_popup_shown)
        self.assertEqual(app.current_tree_preview, "")
        self.assertEqual(app.current_provider, "")
        self.assertEqual(app.current_model, "")
        self.assertEqual(app.tick_var.value, "Tick: -")
        self.assertEqual(app.phase_var.value, "Phase: -")
        self.assertEqual(app.result_var.value, "Result: -")
        self.assertEqual(app.status_var.value, "Scenario failed to build.")
        app._populate_summary_placeholder.assert_called_once_with()
        app._populate_plan_text.assert_called_once_with("", "")
        app._populate_state_text.assert_called_once_with("", "")
        app._update_event_panel.assert_called_once_with([])
        app._draw_placeholder.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
