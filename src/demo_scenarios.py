"""
Reusable demo scenarios for validating the reactive BT runtime.
"""

import argparse
import json
from typing import Dict, List

import py_trees

from .bt_builder import build_tree_from_json
from .robot_actions import RobotWorldState


DYNAMIC_FAILURE_PLAN: List[Dict[str, str]] = [
    {"action": "Pick", "object": "gear"},
    {"action": "MoveTo", "target": "table"},
    {"action": "Place", "object": "gear", "target": "table"},
]


def run_dynamic_failure_demo(drop_location: str = "staging_area") -> py_trees.common.Status:
    """
    Demonstrate that the reactive BT re-issues recovery actions after a fault.
    """

    state = RobotWorldState.from_plan(DYNAMIC_FAILURE_PLAN)
    tree = build_tree_from_json(DYNAMIC_FAILURE_PLAN, world_state=state)
    tree.setup()

    print("[Demo] Scenario: dynamic failure recovery")
    print("[Demo] Plan:")
    print(json.dumps(DYNAMIC_FAILURE_PLAN, indent=2))

    for tick_count in range(1, 8):
        if tick_count == 4:
            dropped_object = state.drop_held_object(drop_location)
            print(
                "\n[Fault] Simulated drop before placement: {} is no longer held and is now at {}.".format(
                    dropped_object or "nothing", drop_location
                )
            )

        tree.tick()
        print("\n[Demo] Tick {}".format(tick_count))
        print("[Demo] Root status: {}".format(tree.root.status.name))
        print(py_trees.display.unicode_tree(tree.root, show_status=True))
        print("[Demo] World state: {}".format(state.summary()))

        if tree.root.status in {
            py_trees.common.Status.SUCCESS,
            py_trees.common.Status.FAILURE,
        }:
            return tree.root.status

    raise RuntimeError("Dynamic failure demo did not terminate within 7 ticks.")


def main() -> None:
    """
    Run one of the reproducible professor-demo scenarios.
    """

    parser = argparse.ArgumentParser(description="Run demo validation scenarios.")
    parser.add_argument(
        "scenario",
        choices=["dynamic_failure"],
        help="Scenario to execute.",
    )
    parser.add_argument(
        "--drop-location",
        default="staging_area",
        help="Location assigned to the dropped object in the dynamic failure demo.",
    )
    args = parser.parse_args()

    if args.scenario == "dynamic_failure":
        final_status = run_dynamic_failure_demo(drop_location=args.drop_location)
        print("\n[Demo] Final status: {}".format(final_status.name))


if __name__ == "__main__":
    main()
