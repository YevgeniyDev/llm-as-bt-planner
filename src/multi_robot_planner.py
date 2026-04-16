"""
MRBTP-inspired multi-robot symbolic backbone for the local BT prototype.

This version keeps the LLM as the semantic front end, but extends the symbolic
team planner with explicit tool changes and explicit cross-robot handoffs.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import py_trees

from .multi_robot_actions import (
    MultiRobotWorldState,
    RobotProfile,
    TeamLiteralCondition,
    build_action_node,
    canonical_action_name,
)


DEFAULT_MULTI_ROBOT_PROFILES: Tuple[RobotProfile, ...] = (
    RobotProfile(name="robot1", priority=0),
    RobotProfile(name="robot2", priority=1),
)
SUPPORTED_TERMINAL_ACTIONS = {"Pick", "MoveTo", "Place", "Insert", "ChangeTool", "Handoff"}


@dataclass(frozen=True)
class PlanSegment:
    index: int
    steps: Tuple[Dict[str, str], ...]
    terminal_action: str
    assigned_robot: Optional[str]
    object_name: Optional[str]
    target: Optional[str]
    tool: Optional[str]
    recipient: Optional[str]
    location: Optional[str]

    @property
    def label(self) -> str:
        if self.terminal_action == "ChangeTool" and self.tool:
            return "ChangeTool({})".format(self.tool)
        if self.terminal_action == "Handoff" and self.object_name and self.recipient:
            if self.location:
                return "Handoff({}, {}, {})".format(
                    self.object_name,
                    self.recipient,
                    self.location,
                )
            return "Handoff({}, {})".format(self.object_name, self.recipient)
        if self.object_name and self.target:
            return "{}({}, {})".format(self.terminal_action, self.object_name, self.target)
        if self.object_name:
            return "{}({})".format(self.terminal_action, self.object_name)
        if self.target:
            return "{}({})".format(self.terminal_action, self.target)
        return self.terminal_action


@dataclass(frozen=True)
class AssignedGoal:
    robot_name: str
    segment: PlanSegment
    goal_literal: str
    primary: bool = True


@dataclass(frozen=True)
class SymbolicAction:
    robot_name: str
    action_name: str
    object_name: Optional[str] = None
    target: Optional[str] = None
    tool: Optional[str] = None
    recipient: Optional[str] = None
    location: Optional[str] = None
    preconditions: frozenset[str] = frozenset()
    add_effects: frozenset[str] = frozenset()
    delete_effects: frozenset[str] = frozenset()

    @property
    def label(self) -> str:
        if self.action_name == "ChangeTool" and self.tool:
            return "ChangeTool({}, {})".format(self.robot_name, self.tool)
        if self.action_name == "Handoff" and self.object_name and self.recipient:
            return "Handoff({}, {}, {}, {})".format(
                self.robot_name,
                self.object_name,
                self.recipient,
                self.location or "",
            )
        if self.object_name and self.target:
            return "{}({}, {}, {})".format(
                self.action_name,
                self.robot_name,
                self.object_name,
                self.target,
            )
        if self.object_name:
            return "{}({}, {})".format(self.action_name, self.robot_name, self.object_name)
        if self.target:
            return "{}({}, {})".format(self.action_name, self.robot_name, self.target)
        return "{}({})".format(self.action_name, self.robot_name)


@dataclass
class PlanningCondition:
    condition_set: frozenset[str]
    action: Optional[SymbolicAction] = None
    children: List["PlanningCondition"] = field(default_factory=list)


class BackwardGoalPlanner:
    def __init__(self, goal_literal: str, start_state: frozenset[str], action_list: Sequence[SymbolicAction]):
        self.goal_literal = goal_literal
        self.start_state = start_state
        self.action_list = list(action_list)
        self.goal_condition = PlanningCondition(frozenset({goal_literal}))
        self.expanded_condition_dict: Dict[frozenset[str], PlanningCondition] = {
            self.goal_condition.condition_set: self.goal_condition
        }

    def plan(self) -> PlanningCondition:
        queue = [self.goal_condition.condition_set]
        visited = set()

        while queue:
            condition = queue.pop(0)
            if condition in visited:
                continue
            visited.add(condition)

            if self.start_state >= condition:
                continue

            new_nodes = self.one_step_expand(condition)
            for node in new_nodes:
                queue.append(node.condition_set)

        return self.goal_condition

    def one_step_expand(self, condition: frozenset[str]) -> List[PlanningCondition]:
        inside_condition = self.expanded_condition_dict.get(condition)
        premise_conditions: List[PlanningCondition] = []

        for action in self.action_list:
            if not self.is_consequence(condition, action):
                continue

            premise_condition = frozenset((action.preconditions | condition) - action.add_effects)
            if not self.has_no_subset(premise_condition):
                continue

            planning_condition = PlanningCondition(premise_condition, action=action)
            premise_conditions.append(planning_condition)
            self.expanded_condition_dict[premise_condition] = planning_condition

        if inside_condition is not None:
            inside_condition.children.extend(premise_conditions)
        elif premise_conditions:
            outside_condition = self.expanded_condition_dict.get(condition)
            if outside_condition is None:
                outside_condition = PlanningCondition(condition)
                self.expanded_condition_dict[condition] = outside_condition
                self.goal_condition.children.append(outside_condition)
            outside_condition.children.extend(premise_conditions)

        return premise_conditions

    def is_consequence(self, condition: frozenset[str], action: SymbolicAction) -> bool:
        if condition & ((action.preconditions | action.add_effects) - action.delete_effects) <= set():
            return False
        if (condition - action.delete_effects) != condition:
            return False
        return True

    def has_no_subset(self, condition: frozenset[str]) -> bool:
        for expanded_condition in self.expanded_condition_dict:
            if expanded_condition <= condition:
                return False
        return True


def build_multi_robot_tree_from_json(
    plan_json: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    robot_profiles: Optional[Sequence[RobotProfile]] = None,
    world_state: Optional[MultiRobotWorldState] = None,
) -> py_trees.trees.BehaviourTree:
    plan = _normalize_plan(plan_json)
    profiles = list(robot_profiles or DEFAULT_MULTI_ROBOT_PROFILES)
    runtime_world_state = world_state or MultiRobotWorldState.from_profiles_and_plan(profiles, plan)
    predicted_world_state = runtime_world_state.clone()

    root = py_trees.composites.Sequence(name="MultiRobotTaskPlan", memory=False)

    phases = group_segments_into_phases(segment_plan(plan))
    for phase_index, phase_segments in enumerate(phases, start=1):
        if len(phase_segments) == 1 and phase_segments[0].terminal_action == "Handoff":
            phase_parallel, assignments = _build_handoff_phase_subtree(
                phase_index=phase_index,
                segment=phase_segments[0],
                robot_profiles=profiles,
                runtime_world_state=runtime_world_state,
                predicted_world_state=predicted_world_state,
            )
        else:
            phase_parallel, assignments = _build_phase_parallel_subtree(
                phase_index=phase_index,
                phase_segments=phase_segments,
                robot_profiles=profiles,
                runtime_world_state=runtime_world_state,
                predicted_world_state=predicted_world_state,
            )

        root.add_child(phase_parallel)
        for assignment in assignments:
            if not assignment.primary:
                continue

            predicted_world_state.apply_predicted_effect(
                assignment.robot_name,
                assignment.segment.terminal_action,
                object_name=assignment.segment.object_name,
                target=assignment.segment.target,
                tool=assignment.segment.tool,
                recipient=assignment.segment.recipient,
                location=assignment.segment.location,
            )

    tree = py_trees.trees.BehaviourTree(root=root)
    tree.world_state = runtime_world_state  # type: ignore[attr-defined]
    tree.robot_profiles = profiles  # type: ignore[attr-defined]
    tree.phase_segments = phases  # type: ignore[attr-defined]
    return tree


def default_multi_robot_profiles() -> List[RobotProfile]:
    return list(DEFAULT_MULTI_ROBOT_PROFILES)


def resolve_robot_profiles(raw_payload: Optional[str] = None) -> List[RobotProfile]:
    if raw_payload is None or not raw_payload.strip():
        return default_multi_robot_profiles()

    payload = json.loads(raw_payload)
    if not isinstance(payload, list) or not payload:
        raise ValueError("MULTI_ROBOT_ROBOTS must be a non-empty JSON array.")

    profiles = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("Each multi-robot profile must be a JSON object.")
        profiles.append(RobotProfile.from_payload(item, default_priority=index))
    return profiles


def segment_plan(plan: Sequence[Dict[str, str]]) -> List[PlanSegment]:
    segments: List[PlanSegment] = []
    current_steps: List[Dict[str, str]] = []
    segment_index = 1

    for step in plan:
        current_steps.append(step)
        action_name = step.get("action")
        if not isinstance(action_name, str):
            continue

        action_key = canonical_action_name(action_name)
        if action_key not in {"Place", "Insert", "Handoff", "ChangeTool"}:
            continue

        segments.append(_make_segment(segment_index, current_steps))
        segment_index += 1
        current_steps = []

    if current_steps:
        segments.append(_make_segment(segment_index, current_steps))

    return segments


def group_segments_into_phases(segments: Sequence[PlanSegment]) -> List[List[PlanSegment]]:
    phases: List[List[PlanSegment]] = []
    current_phase: List[PlanSegment] = []
    used_objects = set()

    for segment in segments:
        if segment.terminal_action == "Handoff":
            if current_phase:
                phases.append(current_phase)
                current_phase = []
                used_objects = set()
            phases.append([segment])
            continue

        object_name = segment.object_name or ""
        if current_phase and object_name and object_name in used_objects:
            phases.append(current_phase)
            current_phase = []
            used_objects = set()

        current_phase.append(segment)
        if object_name:
            used_objects.add(object_name)

    if current_phase:
        phases.append(current_phase)

    return phases


def _make_segment(index: int, steps: Sequence[Dict[str, str]]) -> PlanSegment:
    terminal_step = steps[-1]
    action_name = terminal_step.get("action")
    if not isinstance(action_name, str):
        raise ValueError("Plan segment {} is missing an action.".format(index))

    terminal_action = canonical_action_name(action_name)
    tool = _resolve_segment_tool(steps)
    recipient = _get_optional_field(terminal_step, "recipient", "to")
    location = _get_optional_field(terminal_step, "location", "meeting_point")
    if terminal_action == "Handoff" and location is None:
        location = _get_optional_field(terminal_step, "target", "destination")

    target = None
    if terminal_action != "Handoff":
        target = _get_optional_field(terminal_step, "target", "destination", "location")

    return PlanSegment(
        index=index,
        steps=tuple(dict(step) for step in steps),
        terminal_action=terminal_action,
        assigned_robot=_resolve_segment_robot(steps, index),
        object_name=_get_optional_field(terminal_step, "object", "item"),
        target=target,
        tool=tool,
        recipient=recipient,
        location=location,
    )


def _build_phase_parallel_subtree(
    phase_index: int,
    phase_segments: Sequence[PlanSegment],
    robot_profiles: Sequence[RobotProfile],
    runtime_world_state: MultiRobotWorldState,
    predicted_world_state: MultiRobotWorldState,
) -> Tuple[py_trees.behaviour.Behaviour, List[AssignedGoal]]:
    assignments = allocate_phase_segments(phase_segments, robot_profiles, predicted_world_state)

    if not assignments:
        return py_trees.behaviours.Success(name="Phase {} idle".format(phase_index)), []

    phase_root = py_trees.composites.Parallel(
        name="Phase {}".format(phase_index),
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False),
    )

    assignment_lookup: Dict[str, List[AssignedGoal]] = {}
    for assignment in assignments:
        assignment_lookup.setdefault(assignment.robot_name, []).append(assignment)

    for profile in sorted(robot_profiles, key=lambda item: item.priority):
        robot_assignments = assignment_lookup.get(profile.name, [])
        if not robot_assignments:
            continue

        robot_root = py_trees.composites.Sequence(name="Robot {}".format(profile.name), memory=False)
        for assignment in robot_assignments:
            action_model = build_action_model_for_segment(
                assignment.segment,
                profile,
                predicted_world_state,
            )
            planner = BackwardGoalPlanner(
                goal_literal=assignment.goal_literal,
                start_state=predicted_world_state.to_symbolic_state(),
                action_list=action_model,
            )
            goal_condition = planner.plan()
            robot_root.add_child(
                compile_planning_condition(
                    planning_condition=goal_condition,
                    owner_robot=profile.name,
                    world_state=runtime_world_state,
                    goal_literal=assignment.goal_literal,
                )
            )

        phase_root.add_child(robot_root)

    return phase_root, assignments


def _build_handoff_phase_subtree(
    phase_index: int,
    segment: PlanSegment,
    robot_profiles: Sequence[RobotProfile],
    runtime_world_state: MultiRobotWorldState,
    predicted_world_state: MultiRobotWorldState,
) -> Tuple[py_trees.behaviour.Behaviour, List[AssignedGoal]]:
    recipient_profile = _resolve_recipient_profile(segment, robot_profiles)
    _validate_handoff_support(segment, recipient_profile, predicted_world_state)
    giver_profile = _select_handoff_giver(segment, robot_profiles, predicted_world_state)
    goal_literal = segment_goal_literal(segment, giver_profile.name)

    phase_root = py_trees.composites.Parallel(
        name="Phase {}".format(phase_index),
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False),
    )

    recipient_root = py_trees.composites.Sequence(
        name="Robot {}".format(recipient_profile.name),
        memory=False,
    )
    recipient_root.add_child(
        _build_handoff_support_subtree(segment, recipient_profile, runtime_world_state)
    )
    phase_root.add_child(recipient_root)

    giver_root = py_trees.composites.Sequence(
        name="Robot {}".format(giver_profile.name),
        memory=False,
    )
    action_model = build_action_model_for_segment(segment, giver_profile, predicted_world_state)
    planner = BackwardGoalPlanner(
        goal_literal=goal_literal,
        start_state=predicted_world_state.to_symbolic_state(),
        action_list=action_model,
    )
    goal_condition = planner.plan()
    giver_root.add_child(
        py_trees.decorators.FailureIsRunning(
            name="WaitTo{}".format(goal_literal),
            child=compile_planning_condition(
                planning_condition=goal_condition,
                owner_robot=giver_profile.name,
                world_state=runtime_world_state,
                goal_literal=goal_literal,
            ),
        )
    )
    phase_root.add_child(giver_root)

    return phase_root, [
        AssignedGoal(
            robot_name=giver_profile.name,
            segment=segment,
            goal_literal=goal_literal,
            primary=True,
        )
    ]


def allocate_phase_segments(
    phase_segments: Sequence[PlanSegment],
    robot_profiles: Sequence[RobotProfile],
    predicted_world_state: MultiRobotWorldState,
) -> List[AssignedGoal]:
    if not phase_segments:
        return []

    profiles = sorted(robot_profiles, key=lambda item: item.priority)
    assignments: List[AssignedGoal] = []

    if len(phase_segments) == 1:
        segment = phase_segments[0]
        if segment.assigned_robot:
            return [_assign_explicit_segment(segment, robot_profiles, predicted_world_state)]

        capable_robots = [
            profile
            for profile in profiles
            if robot_can_execute_segment(profile, segment, predicted_world_state)
        ]
        if not capable_robots:
            raise ValueError("No robot can execute segment '{}' in multi-robot mode.".format(segment.label))

        primary_robot = capable_robots[0]
        if _segment_supports_backups(segment):
            for profile in capable_robots:
                assignments.append(
                    AssignedGoal(
                        robot_name=profile.name,
                        segment=segment,
                        goal_literal=segment_goal_literal(segment, profile.name),
                        primary=profile.name == primary_robot.name,
                    )
                )
            return assignments

        assignments.append(
            AssignedGoal(
                robot_name=primary_robot.name,
                segment=segment,
                goal_literal=segment_goal_literal(segment, primary_robot.name),
                primary=True,
            )
        )
        return assignments

    workloads = {profile.name: 0 for profile in profiles}
    for segment in phase_segments:
        if segment.assigned_robot:
            explicit_assignment = _assign_explicit_segment(
                segment,
                robot_profiles,
                predicted_world_state,
            )
            assignments.append(explicit_assignment)
            workloads[explicit_assignment.robot_name] += 1
            continue

        capable_robots = [
            profile
            for profile in profiles
            if robot_can_execute_segment(profile, segment, predicted_world_state)
        ]
        if not capable_robots:
            raise ValueError("No robot can execute segment '{}' in multi-robot mode.".format(segment.label))

        capable_robots.sort(key=lambda profile: (workloads[profile.name], profile.priority))
        primary_robot = capable_robots[0]
        workloads[primary_robot.name] += 1
        assignments.append(
            AssignedGoal(
                robot_name=primary_robot.name,
                segment=segment,
                goal_literal=segment_goal_literal(segment, primary_robot.name),
                primary=True,
            )
        )

    return assignments


def robot_can_execute_segment(
    profile: RobotProfile,
    segment: PlanSegment,
    predicted_world_state: Optional[MultiRobotWorldState] = None,
) -> bool:
    capabilities = {canonical_action_name(capability) for capability in profile.capabilities}
    action_name = segment.terminal_action
    robot_name = profile.name
    object_name = segment.object_name
    required_location = segment.location or segment.target
    required_tool = segment.tool

    object_holder: Optional[str] = None
    already_holding = False
    already_at_target = False
    already_equipped = False
    if predicted_world_state is not None:
        if object_name:
            object_holder = predicted_world_state.find_robot_holding(object_name)
            already_holding = predicted_world_state.is_holding(robot_name, object_name)
        if required_location:
            already_at_target = predicted_world_state.is_at(robot_name, required_location)
        if required_tool:
            already_equipped = predicted_world_state.is_tool_equipped(robot_name, required_tool)

    if object_holder and object_holder != robot_name:
        if action_name in {"Pick", "Place", "Insert"}:
            return False
        if action_name == "Handoff":
            return object_holder == robot_name

    if action_name == "Pick":
        return already_holding or "Pick" in capabilities

    if action_name == "MoveTo":
        return already_at_target or "MoveTo" in capabilities

    if action_name == "ChangeTool":
        return required_tool is not None and _robot_can_satisfy_tool(
            profile,
            required_tool,
            capabilities,
            already_equipped,
        )

    if action_name == "Place":
        return (
            "Place" in capabilities
            and (already_holding or "Pick" in capabilities)
            and (already_at_target or "MoveTo" in capabilities)
        )

    if action_name == "Insert":
        return (
            "Insert" in capabilities
            and (already_holding or "Pick" in capabilities)
            and (already_at_target or "MoveTo" in capabilities)
            and _robot_can_satisfy_tool(profile, required_tool, capabilities, already_equipped)
        )

    if action_name == "Handoff":
        if (
            not segment.object_name
            or not segment.recipient
            or not required_location
            or robot_name == segment.recipient
        ):
            return False
        return (
            "Handoff" in capabilities
            and (already_holding or "Pick" in capabilities)
            and (already_at_target or "MoveTo" in capabilities)
        )

    return False


def segment_goal_literal(segment: PlanSegment, robot_name: str) -> str:
    action_name = segment.terminal_action

    if action_name == "Pick" and segment.object_name:
        return "Holding({}, {})".format(robot_name, segment.object_name)

    if action_name == "MoveTo" and segment.target:
        return "At({}, {})".format(robot_name, segment.target)

    if action_name == "ChangeTool" and segment.tool:
        return "Equipped({}, {})".format(robot_name, segment.tool)

    if action_name == "Place" and segment.object_name and segment.target:
        return "ObjectAt({}, {})".format(segment.object_name, segment.target)

    if action_name == "Insert" and segment.object_name and segment.target:
        return "Inserted({}, {})".format(segment.object_name, segment.target)

    if action_name == "Handoff" and segment.object_name and segment.recipient:
        return "Holding({}, {})".format(segment.recipient, segment.object_name)

    raise ValueError("Unsupported multi-robot segment '{}' for goal conversion.".format(segment.label))


def build_action_model_for_segment(
    segment: PlanSegment,
    profile: RobotProfile,
    predicted_world_state: MultiRobotWorldState,
) -> List[SymbolicAction]:
    robot_name = profile.name
    capabilities = {canonical_action_name(capability) for capability in profile.capabilities}
    actions: List[SymbolicAction] = []
    relevant_location = segment.location or segment.target
    required_tool = segment.tool

    if segment.terminal_action == "Place" and segment.object_name and segment.target:
        actions.append(
            SymbolicAction(
                robot_name=robot_name,
                action_name="Place",
                object_name=segment.object_name,
                target=segment.target,
                preconditions=frozenset(
                    {
                        "Holding({}, {})".format(robot_name, segment.object_name),
                        "At({}, {})".format(robot_name, segment.target),
                    }
                ),
                add_effects=frozenset({"ObjectAt({}, {})".format(segment.object_name, segment.target)}),
                delete_effects=frozenset({"Holding({}, {})".format(robot_name, segment.object_name)}),
            )
        )

    if segment.terminal_action == "Insert" and segment.object_name and segment.target:
        insert_preconditions = {
            "Holding({}, {})".format(robot_name, segment.object_name),
            "At({}, {})".format(robot_name, segment.target),
        }
        if required_tool:
            insert_preconditions.add("Equipped({}, {})".format(robot_name, required_tool))

        actions.append(
            SymbolicAction(
                robot_name=robot_name,
                action_name="Insert",
                object_name=segment.object_name,
                target=segment.target,
                tool=required_tool,
                preconditions=frozenset(insert_preconditions),
                add_effects=frozenset({"Inserted({}, {})".format(segment.object_name, segment.target)}),
                delete_effects=frozenset({"Holding({}, {})".format(robot_name, segment.object_name)}),
            )
        )

    if (
        segment.terminal_action == "Handoff"
        and segment.object_name
        and segment.recipient
        and segment.location
    ):
        actions.append(
            SymbolicAction(
                robot_name=robot_name,
                action_name="Handoff",
                object_name=segment.object_name,
                recipient=segment.recipient,
                location=segment.location,
                preconditions=frozenset(
                    {
                        "Holding({}, {})".format(robot_name, segment.object_name),
                        "At({}, {})".format(robot_name, segment.location),
                        "At({}, {})".format(segment.recipient, segment.location),
                        "HandEmpty({})".format(segment.recipient),
                    }
                ),
                add_effects=frozenset(
                    {
                        "Holding({}, {})".format(segment.recipient, segment.object_name),
                        "HandEmpty({})".format(robot_name),
                    }
                ),
                delete_effects=frozenset(
                    {
                        "Holding({}, {})".format(robot_name, segment.object_name),
                        "HandEmpty({})".format(segment.recipient),
                    }
                ),
            )
        )

    if segment.object_name and "Pick" in capabilities:
        actions.append(
            SymbolicAction(
                robot_name=robot_name,
                action_name="Pick",
                object_name=segment.object_name,
                preconditions=frozenset({"HandEmpty({})".format(robot_name)}),
                add_effects=frozenset({"Holding({}, {})".format(robot_name, segment.object_name)}),
                delete_effects=frozenset({"HandEmpty({})".format(robot_name)}),
            )
        )

    if required_tool and _tool_is_available(profile, required_tool):
        change_tool_action = _build_change_tool_action(robot_name, profile, required_tool)
        if change_tool_action is not None:
            actions.append(change_tool_action)

    if relevant_location and "MoveTo" in capabilities:
        known_locations = _collect_known_locations(segment, predicted_world_state)
        delete_effects = frozenset(
            {
                "At({}, {})".format(robot_name, location)
                for location in known_locations
                if location != relevant_location
            }
        )
        actions.append(
            SymbolicAction(
                robot_name=robot_name,
                action_name="MoveTo",
                target=relevant_location,
                add_effects=frozenset({"At({}, {})".format(robot_name, relevant_location)}),
                delete_effects=delete_effects,
            )
        )

    return actions


def compile_planning_condition(
    planning_condition: PlanningCondition,
    owner_robot: str,
    world_state: MultiRobotWorldState,
    goal_literal: str,
) -> py_trees.behaviour.Behaviour:
    if not planning_condition.children and planning_condition.action is None:
        return build_condition_check(planning_condition.condition_set, owner_robot, world_state)

    selector = py_trees.composites.Selector(
        name="Goal {}".format(
            ", ".join(sorted(planning_condition.condition_set)) or "satisfied"
        ),
        memory=False,
    )
    selector.add_child(build_condition_check(planning_condition.condition_set, owner_robot, world_state))

    for child in planning_condition.children:
        branch = py_trees.composites.Sequence(
            name=child.action.label if child.action else "Recover",
            memory=False,
        )
        branch.add_child(
            compile_planning_condition(
                planning_condition=child,
                owner_robot=owner_robot,
                world_state=world_state,
                goal_literal=goal_literal,
            )
        )
        if child.action is not None:
            branch.add_child(
                build_action_node(
                    robot_name=child.action.robot_name,
                    action_name=child.action.action_name,
                    object_name=child.action.object_name,
                    target=child.action.target,
                    tool=child.action.tool,
                    recipient=child.action.recipient,
                    location=child.action.location,
                    world_state=world_state,
                    goal_literal=goal_literal,
                )
            )
        selector.add_child(branch)

    return selector


def build_condition_check(
    condition_set: Iterable[str],
    owner_robot: str,
    world_state: MultiRobotWorldState,
) -> py_trees.behaviour.Behaviour:
    literals = sorted(condition_set)
    if not literals:
        return py_trees.behaviours.Success(name="Satisfied")

    if len(literals) == 1:
        return TeamLiteralCondition(literals[0], world_state, owner_robot)

    sequence = py_trees.composites.Sequence(
        name="Check {}".format(" & ".join(literals)),
        memory=False,
    )
    for literal in literals:
        sequence.add_child(TeamLiteralCondition(literal, world_state, owner_robot))
    return sequence


def _build_handoff_support_subtree(
    segment: PlanSegment,
    recipient_profile: RobotProfile,
    world_state: MultiRobotWorldState,
) -> py_trees.behaviour.Behaviour:
    if not segment.location:
        raise ValueError("Handoff segment '{}' is missing a location.".format(segment.label))

    recipient_name = recipient_profile.name
    support_root = py_trees.composites.Selector(
        name="Support {}".format(segment.label),
        memory=False,
    )

    if segment.object_name:
        support_root.add_child(
            TeamLiteralCondition(
                "Holding({}, {})".format(recipient_name, segment.object_name),
                world_state,
                recipient_name,
            )
        )

    prepare_receive = py_trees.composites.Sequence(
        name="Prepare {}".format(segment.label),
        memory=False,
    )
    prepare_receive.add_child(
        TeamLiteralCondition(
            "HandEmpty({})".format(recipient_name),
            world_state,
            recipient_name,
        )
    )

    ensure_at = py_trees.composites.Selector(
        name="EnsureAt({}, {})".format(recipient_name, segment.location),
        memory=False,
    )
    ensure_at.add_child(
        TeamLiteralCondition(
            "At({}, {})".format(recipient_name, segment.location),
            world_state,
            recipient_name,
        )
    )
    ensure_at.add_child(
        build_action_node(
            robot_name=recipient_name,
            action_name="MoveTo",
            target=segment.location,
            world_state=world_state,
            goal_literal="At({}, {})".format(recipient_name, segment.location),
        )
    )
    prepare_receive.add_child(ensure_at)
    support_root.add_child(prepare_receive)
    return support_root


def _segment_supports_backups(segment: PlanSegment) -> bool:
    return segment.terminal_action in {"Place", "Insert"}


def _resolve_recipient_profile(
    segment: PlanSegment,
    robot_profiles: Sequence[RobotProfile],
) -> RobotProfile:
    if not segment.recipient:
        raise ValueError("Handoff segment '{}' is missing a recipient.".format(segment.label))

    for profile in robot_profiles:
        if profile.name == segment.recipient:
            return profile

    raise ValueError(
        "Handoff recipient '{}' is not part of the configured robot team.".format(
            segment.recipient
        )
    )


def _validate_handoff_support(
    segment: PlanSegment,
    recipient_profile: RobotProfile,
    predicted_world_state: MultiRobotWorldState,
) -> None:
    if not segment.location:
        raise ValueError("Handoff segment '{}' is missing a location.".format(segment.label))

    if predicted_world_state.held_objects.get(recipient_profile.name) is not None:
        raise ValueError(
            "Recipient '{}' is already holding an object, so '{}' cannot start.".format(
                recipient_profile.name,
                segment.label,
            )
        )

    recipient_capabilities = {
        canonical_action_name(capability) for capability in recipient_profile.capabilities
    }
    if (
        not predicted_world_state.is_at(recipient_profile.name, segment.location)
        and "MoveTo" not in recipient_capabilities
    ):
        raise ValueError(
            "Recipient '{}' cannot reach handoff location '{}' because MoveTo is unavailable.".format(
                recipient_profile.name,
                segment.location,
            )
        )


def _select_handoff_giver(
    segment: PlanSegment,
    robot_profiles: Sequence[RobotProfile],
    predicted_world_state: MultiRobotWorldState,
) -> RobotProfile:
    if segment.assigned_robot:
        for profile in robot_profiles:
            if profile.name != segment.assigned_robot:
                continue
            if not robot_can_execute_segment(profile, segment, predicted_world_state):
                raise ValueError(
                    "Assigned handoff giver '{}' cannot execute '{}'.".format(
                        segment.assigned_robot,
                        segment.label,
                    )
                )
            return profile

        raise ValueError(
            "Assigned handoff giver '{}' is not part of the configured robot team.".format(
                segment.assigned_robot
            )
        )

    capable_profiles = [
        profile
        for profile in sorted(robot_profiles, key=lambda item: item.priority)
        if robot_can_execute_segment(profile, segment, predicted_world_state)
    ]
    if not capable_profiles:
        raise ValueError("No robot can execute handoff segment '{}'.".format(segment.label))

    if segment.object_name:
        current_holder = predicted_world_state.find_robot_holding(segment.object_name)
        if current_holder:
            for profile in capable_profiles:
                if profile.name == current_holder:
                    return profile

    return capable_profiles[0]


def _robot_can_satisfy_tool(
    profile: RobotProfile,
    tool_name: Optional[str],
    capabilities: set[str],
    already_equipped: bool,
) -> bool:
    if not tool_name:
        return True
    if already_equipped:
        return True
    if not _tool_is_available(profile, tool_name):
        return False
    return "ChangeTool" in capabilities or profile.default_tool == tool_name


def _tool_is_available(profile: RobotProfile, tool_name: str) -> bool:
    cleaned_tool = tool_name.strip()
    if not cleaned_tool:
        return False

    available_tools = set(profile.available_tools)
    if not available_tools:
        available_tools = {profile.default_tool}
    return cleaned_tool in available_tools


def _build_change_tool_action(
    robot_name: str,
    profile: RobotProfile,
    tool_name: str,
) -> Optional[SymbolicAction]:
    if not _tool_is_available(profile, tool_name):
        return None

    delete_effects = frozenset(
        "Equipped({}, {})".format(robot_name, other_tool)
        for other_tool in profile.available_tools
        if other_tool != tool_name
    )
    return SymbolicAction(
        robot_name=robot_name,
        action_name="ChangeTool",
        tool=tool_name,
        add_effects=frozenset({"Equipped({}, {})".format(robot_name, tool_name)}),
        delete_effects=delete_effects,
    )


def _collect_known_locations(
    segment: PlanSegment,
    predicted_world_state: MultiRobotWorldState,
) -> List[str]:
    known_locations = set(predicted_world_state.robot_locations.values())
    if segment.target:
        known_locations.add(segment.target)
    if segment.location:
        known_locations.add(segment.location)

    for step in segment.steps:
        location = _get_optional_field(step, "target", "destination", "location")
        if location:
            known_locations.add(location)

    return sorted(known_locations)


def _normalize_plan(plan_json: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    payload: Any = plan_json
    if isinstance(plan_json, str):
        payload = json.loads(plan_json)

    if isinstance(payload, dict):
        if "plan" in payload:
            payload = payload["plan"]
        elif "steps" in payload:
            payload = payload["steps"]

    if not isinstance(payload, list) or not payload:
        raise ValueError("Plan JSON must resolve to a non-empty list of steps.")

    return payload


def _resolve_segment_tool(steps: Sequence[Dict[str, str]]) -> Optional[str]:
    for step in reversed(steps):
        action_name = step.get("action")
        if isinstance(action_name, str) and canonical_action_name(action_name) == "ChangeTool":
            tool_name = _get_optional_field(step, "tool", "target", "object")
            if tool_name:
                return tool_name

        tool_name = _get_optional_field(step, "tool")
        if tool_name:
            return tool_name

    return None


def _get_optional_field(step: Dict[str, Any], *field_names: str) -> Optional[str]:
    for field_name in field_names:
        value = step.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_segment_robot(
    steps: Sequence[Dict[str, str]],
    segment_index: int,
) -> Optional[str]:
    robot_names = {
        value.strip()
        for step in steps
        for value in [step.get("robot")]
        if isinstance(value, str) and value.strip()
    }
    if not robot_names:
        return None
    if len(robot_names) > 1:
        raise ValueError(
            "Plan segment {} mixes multiple explicit robots: {}.".format(
                segment_index,
                ", ".join(sorted(robot_names)),
            )
        )
    return next(iter(robot_names))


def _assign_explicit_segment(
    segment: PlanSegment,
    robot_profiles: Sequence[RobotProfile],
    predicted_world_state: MultiRobotWorldState,
) -> AssignedGoal:
    if not segment.assigned_robot:
        raise ValueError("Explicit segment assignment requires an assigned_robot.")

    for profile in robot_profiles:
        if profile.name != segment.assigned_robot:
            continue

        if not robot_can_execute_segment(profile, segment, predicted_world_state):
            raise ValueError(
                "Assigned robot '{}' cannot execute segment '{}'.".format(
                    segment.assigned_robot,
                    segment.label,
                )
            )

        return AssignedGoal(
            robot_name=profile.name,
            segment=segment,
            goal_literal=segment_goal_literal(segment, profile.name),
            primary=True,
        )

    raise ValueError(
        "Assigned robot '{}' is not part of the configured robot team.".format(
            segment.assigned_robot
        )
    )
