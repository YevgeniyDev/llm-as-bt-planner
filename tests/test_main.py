import unittest
from unittest import mock

from src import main as main_module


class MainInstructionResolutionTests(unittest.TestCase):
    def test_cli_arguments_override_terminal_prompt(self):
        with mock.patch.object(main_module.sys, "argv", ["main.py", "Assemble", "the", "gearbox."]):
            self.assertEqual(main_module.resolve_instruction(), "Assemble the gearbox.")

    def test_interactive_terminal_uses_typed_instruction(self):
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = True

        with mock.patch.object(main_module.sys, "argv", ["main.py"]):
            with mock.patch.object(main_module.sys, "stdin", fake_stdin):
                with mock.patch("builtins.input", return_value="Insert the gear into the chassis."):
                    self.assertEqual(
                        main_module.resolve_instruction(),
                        "Insert the gear into the chassis.",
                    )

    def test_interactive_terminal_uses_default_when_input_is_empty(self):
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = True

        with mock.patch.object(main_module.sys, "argv", ["main.py"]):
            with mock.patch.object(main_module.sys, "stdin", fake_stdin):
                with mock.patch("builtins.input", return_value=""):
                    self.assertEqual(
                        main_module.resolve_instruction(),
                        main_module.DEFAULT_INSTRUCTION,
                    )

    def test_non_interactive_run_uses_default_instruction(self):
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = False

        with mock.patch.object(main_module.sys, "argv", ["main.py"]):
            with mock.patch.object(main_module.sys, "stdin", fake_stdin):
                self.assertEqual(
                    main_module.resolve_instruction(),
                    main_module.DEFAULT_INSTRUCTION,
                )


if __name__ == "__main__":
    unittest.main()
