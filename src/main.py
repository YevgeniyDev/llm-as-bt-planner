"""
Entry point for the LLM-as-BT planner prototype.

Running this module demonstrates the upgraded pipeline:
1. Load credentials from `.env`
2. Ask the LLM for a symbolic task plan
3. Compile the plan into a reactive behavior tree with conditions and fallbacks
4. Optionally run a human-in-the-loop review round before execution
5. Tick the tree until it succeeds or fails
"""

import json
import os
import sys
import time
from typing import Callable, List, Optional, Tuple

import py_trees
from dotenv import load_dotenv

from .bt_builder import build_tree_from_json
from .llm_client import LLMTaskPlanner
from .multi_robot_planner import build_multi_robot_tree_from_json, resolve_robot_profiles
from .plan_validator import validate_reactive_plan
from .recursive_planner import RecursiveBTPlanner, render_recursive_trace


DEFAULT_INSTRUCTION = """Pick up the screwdriver. Move to the panel.
Place the screwdriver on the panel. Pick up the hammer. Move to the table.
Place the hammer on the table. Pick up the gear. Move to the chassis.
Insert the gear into the chassis."""


TreeBuilder = Callable[[List[dict]], py_trees.trees.BehaviourTree]


def resolve_instruction() -> str:
    """
    Use a CLI-provided instruction when available. Otherwise, prompt in an
    interactive terminal and fall back to the default example on empty input.
    """

    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    if sys.stdin.isatty():
        print("[Main] Enter a task instruction and press Enter.")
        print("[Main] Leave it empty to use the default example:")
        print(DEFAULT_INSTRUCTION)
        try:
            typed_instruction = input("\n[Main] Instruction: ").strip()
        except EOFError:
            return DEFAULT_INSTRUCTION

        if typed_instruction:
            return typed_instruction

    return DEFAULT_INSTRUCTION


def resolve_planning_scheme() -> str:
    """
    Choose which paper-inspired planning scheme to run.
    """

    raw_value = os.getenv("PLANNING_SCHEME", "scheme3").strip().lower()
    if raw_value in {"scheme3", "human_in_the_loop", "human"}:
        return "scheme3"
    if raw_value in {"scheme4", "recursive"}:
        return "scheme4"

    raise ValueError(
        "Unsupported PLANNING_SCHEME '{}'. Use 'scheme3' or 'scheme4'.".format(raw_value)
    )


def should_use_multi_robot() -> bool:
    raw_value = os.getenv("ENABLE_MULTI_ROBOT", "false").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def resolve_tree_builder() -> Tuple[TreeBuilder, Optional[list]]:
    if not should_use_multi_robot():
        return build_tree_from_json, None

    robot_profiles = resolve_robot_profiles(os.getenv("MULTI_ROBOT_ROBOTS"))
    return (
        lambda plan: build_multi_robot_tree_from_json(plan, robot_profiles=robot_profiles),
        robot_profiles,
    )


def execute_tree(
    tree: py_trees.trees.BehaviourTree,
    tick_period_seconds: float = 0.5,
    max_ticks: int = 30,
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


def render_tree(tree: py_trees.trees.BehaviourTree) -> str:
    """
    Render a readable preview of the reactive BT for demos and review rounds.
    """

    return py_trees.display.unicode_tree(root=tree.root, show_status=False)


def print_plan_validation_warnings(plan: List[dict]) -> List[str]:
    """
    Print pre-execution warnings for plans that may oscillate reactively.
    """

    warnings = validate_reactive_plan(plan)
    if not warnings:
        return []

    print("\n[Validate] Reactive plan warnings:")
    for warning in warnings:
        print("[Validate] - {}".format(warning))

    return warnings


def should_run_human_review() -> bool:
    """
    Enable Scheme 3 review only when the session is interactive and not disabled.
    """

    raw_value = os.getenv("ENABLE_HUMAN_IN_THE_LOOP", "true").strip().lower()
    return raw_value not in {"0", "false", "no"} and sys.stdin.isatty()


def ask_human_review() -> Tuple[bool, str]:
    """
    Ask the operator to approve or critique the compiled BT before execution.
    """

    while True:
        answer = input("\n[Review] Does this reactive BT look correct? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True, ""

        if answer in {"n", "no"}:
            feedback = input("[Review] What should the planner fix?: ").strip()
            if feedback:
                return False, feedback
            return (
                False,
                "The human reviewer rejected the tree. Revise the plan so it better matches the instruction.",
            )

        print("[Review] Please answer with 'y' or 'n'.")


def review_plan_with_human(
    planner: LLMTaskPlanner,
    instruction: str,
    plan: List[dict],
    tree_builder: TreeBuilder = build_tree_from_json,
) -> Tuple[List[dict], py_trees.trees.BehaviourTree]:
    """
    Run Scheme 3 human-in-the-loop review and allow iterative plan repair.
    """

    current_plan = plan
    max_rounds = max(1, int(os.getenv("MAX_REVIEW_ROUNDS", "3")))

    for round_index in range(1, max_rounds + 1):
        tree = tree_builder(current_plan)
        tree_preview = render_tree(tree)

        print("\n[Review] Round {}/{}".format(round_index, max_rounds))
        print_plan_validation_warnings(current_plan)
        print("[BT] Reactive tree preview:")
        print(tree_preview)

        approved, feedback = ask_human_review()
        if approved:
            return current_plan, tree

        print("\n[Review] Revising plan from human feedback...")
        current_plan = planner.revise_plan(
            instruction=instruction,
            current_plan=current_plan,
            human_feedback=feedback,
            tree_preview=tree_preview,
        )
        print("\n[LLM] Revised JSON plan:")
        print(json.dumps(current_plan, indent=2))

    raise RuntimeError(
        "The plan was rejected in all {} human review rounds.".format(max_rounds)
    )


def main() -> None:
    """Run the complete instruction-to-behavior-tree pipeline."""

    load_dotenv()

    instruction = resolve_instruction()
    planning_scheme = resolve_planning_scheme()
    tree_builder, robot_profiles = resolve_tree_builder()
    print("[Main] Instruction: {}".format(instruction))
    print("[Main] Planning scheme: {}".format(planning_scheme))

    if robot_profiles is not None:
        robot_summary = ", ".join(
            "{}<{}>".format(profile.name, "/".join(profile.capabilities))
            for profile in robot_profiles
        )
        print("[Main] Multi-robot mode: enabled")
        print("[Main] Robot team: {}".format(robot_summary))

    planner = LLMTaskPlanner()
    print("[Main] Using provider: {}".format(planner.provider))
    print("[Main] Using model: {}".format(planner.model))

    if planning_scheme == "scheme4":
        recursive_planner = RecursiveBTPlanner(
            planner=planner,
            max_depth=max(1, int(os.getenv("MAX_RECURSION_DEPTH", "3"))),
            max_subgoals_per_level=max(1, int(os.getenv("MAX_SUBGOALS_PER_LEVEL", "4"))),
        )
        recursive_trace = recursive_planner.make_tree(instruction)
        plan = recursive_trace.plan

        print("\n[Recursive] Algorithm 1 planning trace:")
        print(render_recursive_trace(recursive_trace))

        print("\n[LLM] Recursive flat JSON plan:")
        print(json.dumps(plan, indent=2))
        print_plan_validation_warnings(plan)

        if os.getenv("ENABLE_HUMAN_IN_THE_LOOP", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }:
            print("\n[Review] Scheme 4 selected, so Scheme 3 human review is skipped for this run.")

        tree = tree_builder(plan)
        print("\n[BT] Reactive tree preview:")
        print(render_tree(tree))
    else:
        plan = planner.plan_task(instruction)

        print("\n[LLM] Generated JSON plan:")
        print(json.dumps(plan, indent=2))

        if should_run_human_review():
            plan, tree = review_plan_with_human(
                planner,
                instruction,
                plan,
                tree_builder=tree_builder,
            )
        else:
            print_plan_validation_warnings(plan)
            if os.getenv("ENABLE_HUMAN_IN_THE_LOOP", "true").strip().lower() not in {
                "0",
                "false",
                "no",
            }:
                print("\n[Review] Human-in-the-loop review skipped because stdin is not interactive.")

            tree = tree_builder(plan)
            print("\n[BT] Reactive tree preview:")
            print(render_tree(tree))

    final_status = execute_tree(tree)

    print("\n[Main] Final behavior tree status: {}".format(final_status.name))

    world_state = getattr(tree, "world_state", None)
    if world_state is not None:
        print("[Main] Final world state: {}".format(world_state.summary()))


if __name__ == "__main__":
    main()
