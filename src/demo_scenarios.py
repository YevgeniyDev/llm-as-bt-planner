"""
Reusable demo scenarios for validating the reactive BT runtime.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

import py_trees
from dotenv import load_dotenv

from .bt_builder import build_tree_from_json
from .gridworld_env import (
    build_env_from_typed_scenario,
    prompt_for_typed_scenario,
    resolve_typed_scenario_text,
)
from .llm_client import LLMTaskPlanner
from .multi_robot_actions import RobotProfile
from .multi_robot_planner import build_multi_robot_tree_from_json
from .robot_actions import RobotWorldState


DYNAMIC_FAILURE_PLAN: List[Dict[str, str]] = [
    {"action": "Pick", "object": "gear"},
    {"action": "MoveTo", "target": "table"},
    {"action": "Place", "object": "gear", "target": "table"},
]

MULTI_ROBOT_PARALLEL_PLAN: List[Dict[str, str]] = [
    {"action": "Pick", "object": "screwdriver"},
    {"action": "MoveTo", "target": "panel"},
    {"action": "Place", "object": "screwdriver", "target": "panel"},
    {"action": "Pick", "object": "hammer"},
    {"action": "MoveTo", "target": "table"},
    {"action": "Place", "object": "hammer", "target": "table"},
]

HETEROGENEOUS_HANDOFF_PLAN: List[Dict[str, str]] = [
    {"action": "Pick", "object": "gear"},
    {"action": "MoveTo", "target": "handoff_station"},
    {"action": "Handoff", "object": "gear", "recipient": "robot2", "location": "handoff_station"},
    {"action": "ChangeTool", "tool": "precision_gripper"},
    {"action": "MoveTo", "target": "chassis"},
    {"action": "Insert", "object": "gear", "target": "chassis", "tool": "precision_gripper"},
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


def run_multi_robot_parallel_demo() -> py_trees.common.Status:
    """
    Demonstrate that two robots can execute distinct goal segments in parallel.
    """

    profiles = [
        RobotProfile(name="robot1", priority=0),
        RobotProfile(name="robot2", priority=1),
    ]
    tree = build_multi_robot_tree_from_json(MULTI_ROBOT_PARALLEL_PLAN, robot_profiles=profiles)
    tree.setup()

    print("[Demo] Scenario: multi-robot parallel execution")
    print("[Demo] Plan:")
    print(json.dumps(MULTI_ROBOT_PARALLEL_PLAN, indent=2))

    for tick_count in range(1, 8):
        tree.tick()
        print("\n[Demo] Tick {}".format(tick_count))
        print("[Demo] Root status: {}".format(tree.root.status.name))
        print(py_trees.display.unicode_tree(tree.root, show_status=True))
        print("[Demo] World state: {}".format(tree.world_state.summary()))

        if tree.root.status in {
            py_trees.common.Status.SUCCESS,
            py_trees.common.Status.FAILURE,
        }:
            return tree.root.status

    raise RuntimeError("Multi-robot parallel demo did not terminate within 7 ticks.")


def run_heterogeneous_handoff_demo() -> py_trees.common.Status:
    """
    Demonstrate helper-to-specialist collaboration with handoff and tool change.
    """

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
    tree = build_multi_robot_tree_from_json(HETEROGENEOUS_HANDOFF_PLAN, robot_profiles=profiles)
    tree.setup()

    print("[Demo] Scenario: heterogeneous handoff and tool-change collaboration")
    print("[Demo] Plan:")
    print(json.dumps(HETEROGENEOUS_HANDOFF_PLAN, indent=2))

    for tick_count in range(1, 13):
        tree.tick()
        print("\n[Demo] Tick {}".format(tick_count))
        print("[Demo] Root status: {}".format(tree.root.status.name))
        print(py_trees.display.unicode_tree(tree.root, show_status=True))
        print("[Demo] World state: {}".format(tree.world_state.summary()))

        if tree.root.status in {
            py_trees.common.Status.SUCCESS,
            py_trees.common.Status.FAILURE,
        }:
            return tree.root.status

    raise RuntimeError("Heterogeneous handoff demo did not terminate within 12 ticks.")


def run_typed_gridworld_demo(
    scenario_text: Optional[str] = None,
    num_robots: int = 3,
    num_circles: Optional[int] = None,
    layout_name: str = "open_room",
    seed: int = 0,
    max_steps: int = 70,
    text_only: bool = False,
) -> bool:
    """
    Run the local MRBTP-inspired gridworld harness from a typed scenario.
    """

    if not text_only:
        try:
            from .gridworld_app import launch_gridworld_tester

            visual_result = launch_gridworld_tester(
                scenario_text=scenario_text,
                num_robots=num_robots,
                num_circles=num_circles,
                layout_name=layout_name,
                seed=seed,
                max_steps=max_steps,
            )
            return bool(visual_result)
        except Exception as error:
            print("[GridWorld] Visual tester unavailable: {}".format(error))
            print("[GridWorld] Falling back to text mode.")

    load_dotenv()
    resolved_text = resolve_typed_scenario_text(scenario_text)
    planner = LLMTaskPlanner()
    env = build_env_from_typed_scenario(
        resolved_text,
        num_robots=num_robots,
        num_circles=num_circles,
        layout_name=layout_name,
        seed=seed,
        planner=planner,
    )
    initial_summary = env.describe()
    tree = env.build_behavior_tree()
    result = env.run(max_steps=max_steps)

    print("[GridWorld] Using provider: {}".format(planner.provider))
    print("[GridWorld] Using model: {}".format(planner.model))
    print("[GridWorld] Scenario:")
    print(result.scenario.raw_text)
    print("[GridWorld] LLM scenario summary:")
    print(result.scenario.task_summary)
    print("[GridWorld] Initial environment:")
    print(initial_summary)
    print("[GridWorld] Derived symbolic plan:")
    print(json.dumps(env.build_symbolic_plan(), indent=2))
    print("[GridWorld] MRBTP-style BT preview:")
    print(py_trees.display.unicode_tree(tree.root, show_status=False))

    for snapshot in result.history:
        print("\n[GridWorld] Tick {}".format(snapshot.tick))
        for event in snapshot.events:
            print(event)
        print(snapshot.render)

    print(
        "\n[GridWorld] Final status: {}".format(
            "SUCCESS" if result.completed else "INCOMPLETE"
        )
    )
    print("[GridWorld] Steps run: {}".format(result.steps_run))
    print("[GridWorld] Final environment:")
    print(env.describe())
    return result.completed


def main() -> None:
    """
    Run one of the reproducible professor-demo scenarios.
    """

    parser = argparse.ArgumentParser(description="Run demo validation scenarios.")
    parser.add_argument(
        "scenario",
        choices=[
            "dynamic_failure",
            "multi_robot_parallel",
            "heterogeneous_handoff",
            "typed_gridworld",
        ],
        help="Scenario to execute.",
    )
    parser.add_argument(
        "--drop-location",
        default="staging_area",
        help="Location assigned to the dropped object in the dynamic failure demo.",
    )
    parser.add_argument(
        "--scenario-text",
        default=None,
        help="Typed gridworld scenario text. Only used by the typed_gridworld demo.",
    )
    parser.add_argument(
        "--num-robots",
        type=int,
        default=3,
        help="Robot count for the typed_gridworld demo.",
    )
    parser.add_argument(
        "--num-objects",
        "--num-circles",
        type=int,
        dest="num_circles",
        default=None,
        help="Optional exact movable-object count for the typed_gridworld demo.",
    )
    parser.add_argument(
        "--layout",
        choices=["open_room", "split_room", "four_rooms", "handoff_hall"],
        default="open_room",
        help="Grid layout for the typed_gridworld demo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the typed_gridworld demo.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=70,
        help="Maximum number of simulation steps for the typed_gridworld demo.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run the typed_gridworld demo in terminal mode instead of the visual window.",
    )
    args = parser.parse_args()

    if args.scenario == "dynamic_failure":
        final_status = run_dynamic_failure_demo(drop_location=args.drop_location)
    elif args.scenario == "heterogeneous_handoff":
        final_status = run_heterogeneous_handoff_demo()
    elif args.scenario == "typed_gridworld":
        scenario_text = args.scenario_text
        if scenario_text is None and args.text_only and sys.stdin.isatty():
            scenario_text = prompt_for_typed_scenario()

        completed = run_typed_gridworld_demo(
            scenario_text=scenario_text,
            num_robots=max(1, args.num_robots),
            num_circles=max(1, args.num_circles) if args.num_circles is not None else None,
            layout_name=args.layout,
            seed=args.seed,
            max_steps=max(1, args.max_steps),
            text_only=args.text_only,
        )
        final_status = (
            py_trees.common.Status.SUCCESS
            if completed
            else py_trees.common.Status.FAILURE
        )
    else:
        final_status = run_multi_robot_parallel_demo()

    print("\n[Demo] Final status: {}".format(final_status.name))


if __name__ == "__main__":
    main()
