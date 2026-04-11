"""
Entry point for the LLM-as-BT planner prototype.

Running this module demonstrates the full pipeline:
1. Load credentials from `.env`
2. Ask the LLM for a symbolic task plan
3. Convert the plan into a behavior tree
4. Tick the tree until it succeeds or fails
"""

import json
import sys
import time

import py_trees
from dotenv import load_dotenv

from .bt_builder import build_tree_from_json
from .llm_client import LLMTaskPlanner


DEFAULT_INSTRUCTION = '''Pick up the screwdriver. Move to the panel. 
Place the screwdriver on the panel. Pick up the hammer. Move to the table. 
Place the hammer on the table. Pick up the gear. Move to the chassis. 
Insert the gear into the chassis.'''


def resolve_instruction() -> str:
    """
    Use a CLI-provided instruction when available, otherwise fall back to the
    example requested in the project brief.
    """

    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    return DEFAULT_INSTRUCTION


def execute_tree(
    tree: py_trees.trees.BehaviourTree,
    tick_period_seconds: float = 0.5,
    max_ticks: int = 20,
) -> py_trees.common.Status:
    """
    Tick the tree until it reaches a terminal state.

    The explicit loop is pedagogically useful for this repository because it
    exposes how behavior trees progress over time instead of hiding execution in
    a convenience helper.
    """

    tree.setup()

    for tick_count in range(1, max_ticks + 1):
        print("\n[BT] Tick {}".format(tick_count))
        tree.tick()

        root_status = tree.root.status
        print("[BT] Root status: {}".format(root_status.name))

        if root_status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            return root_status

        time.sleep(tick_period_seconds)

    raise RuntimeError(
        "Behavior tree did not finish after {} ticks. Check for a non-terminating action.".format(
            max_ticks
        )
    )


def main() -> None:
    """Run the complete instruction-to-behavior-tree pipeline."""

    load_dotenv()

    instruction = resolve_instruction()
    print("[Main] Instruction: {}".format(instruction))

    planner = LLMTaskPlanner()
    print("[Main] Using provider: {}".format(planner.provider))
    print("[Main] Using model: {}".format(planner.model))

    plan = planner.plan_task(instruction)

    print("\n[LLM] Generated JSON plan:")
    print(json.dumps(plan, indent=2))

    tree = build_tree_from_json(plan)
    final_status = execute_tree(tree)

    print("\n[Main] Final behavior tree status: {}".format(final_status.name))


if __name__ == "__main__":
    main()
