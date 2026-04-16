"""
LLM-driven gridworld harness inspired by the MRBTP MiniGrid workflow.

The key change from the earlier local parser prototype is that typed scenarios
now go through the repository's LLM front end first. The model produces a
structured multi-robot scenario spec, the MRBTP-style planner can compile the
result into a team BT, and this module executes the same authored plan inside a
lightweight ASCII gridworld simulator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import py_trees

from .gridworld_domain import (
    DEFAULT_TOOL_NAME,
    FIXTURE_LOCATIONS,
    SUPPORTED_OBJECT_KINDS,
    SUPPORTED_RENDER_SHAPES,
    TOOL_STATION_LOCATIONS,
    default_render_shape,
    infer_object_kind,
    insert_targets_for_kind,
    is_insert_compatible,
)
from .gridworld_layouts import GridLayout, Position, layout_registry
from .llm_client import LLMTaskPlanner
from .multi_robot_actions import RobotProfile, canonical_action_name
from .multi_robot_planner import (
    build_multi_robot_tree_from_json,
    group_segments_into_phases,
    segment_plan,
)


DEFAULT_TYPED_SCENARIO = "All robots collect circles and place them in different corners of the room"


@dataclass(frozen=True)
class GridWorldRobotDirective:
    name: str
    role: str
    start_location: str
    can_move: bool = True


@dataclass(frozen=True)
class GridWorldObjectDirective:
    name: str
    kind: str
    shape: str


@dataclass(frozen=True)
class GridWorldSuccessCondition:
    object_name: str
    target: str


@dataclass(frozen=True)
class TypedTransportScenario:
    raw_text: str
    task_summary: str
    robots: Tuple[GridWorldRobotDirective, ...]
    objects: Tuple[GridWorldObjectDirective, ...]
    plan: Tuple[Dict[str, str], ...]
    success_conditions: Tuple[GridWorldSuccessCondition, ...]


@dataclass
class GridObject:
    name: str
    kind: str
    shape: str
    position: Position
    delivered: bool = False
    held_by: Optional[str] = None
    inserted_target: Optional[str] = None


@dataclass
class GridRobot:
    name: str
    role: str
    position: Position
    can_move: bool = True
    current_tool: str = DEFAULT_TOOL_NAME
    carrying: Optional[str] = None
    next_step_index: int = 0


@dataclass(frozen=True)
class GridWorldFrameRobot:
    name: str
    role: str
    position: Position
    can_move: bool
    carrying: Optional[str]
    current_tool: str


@dataclass(frozen=True)
class GridWorldFrameObject:
    name: str
    kind: str
    shape: str
    position: Position
    delivered: bool
    held_by: Optional[str]
    inserted_target: Optional[str]


@dataclass(frozen=True)
class GridWorldFrameLocation:
    name: str
    category: str
    position: Position


@dataclass(frozen=True)
class GridWorldFrame:
    width: int
    height: int
    walls: Tuple[Position, ...]
    goals: Tuple[Tuple[str, str, Position], ...]
    fixed_locations: Tuple[GridWorldFrameLocation, ...]
    robots: Tuple[GridWorldFrameRobot, ...]
    objects: Tuple[GridWorldFrameObject, ...]
    current_phase_index: int
    total_phases: int


@dataclass
class StepSnapshot:
    tick: int
    render: str
    events: Tuple[str, ...]
    frame: GridWorldFrame


@dataclass
class SimulationResult:
    scenario: TypedTransportScenario
    steps_run: int
    completed: bool
    history: List[StepSnapshot] = field(default_factory=list)


class TypedGridWorldEnv:
    """
    Lightweight multi-robot simulator driven by an LLM-authored execution spec.
    """

    def __init__(
        self,
        scenario: TypedTransportScenario,
        layout: Optional[GridLayout] = None,
        seed: int = 0,
    ) -> None:
        self.scenario = scenario
        self.layout = layout or layout_registry()["open_room"]
        self.random = random.Random(seed)
        self.width = self.layout.width
        self.height = self.layout.height
        self.walls = set(self.layout.walls)
        self.named_locations = named_layout_locations(self.layout)
        self.success_targets = {
            condition.object_name: condition.target
            for condition in self.scenario.success_conditions
        }
        self.phase_plans = self._group_plan_by_phase(self.scenario.plan)
        self.current_phase_index = 0
        self.robot_display = {
            directive.name: str((index % 10) or 0)
            for index, directive in enumerate(self.scenario.robots, start=1)
        }

        self.robots = self._spawn_robots()
        self.objects = self._spawn_objects()
        self._refresh_delivery_flags()

    def all_objects_delivered(self) -> bool:
        return all(
            self._is_success_condition_met(condition)
            for condition in self.scenario.success_conditions
        )

    def build_symbolic_plan(self) -> List[Dict[str, str]]:
        return [dict(step) for step in self.scenario.plan]

    def build_robot_profiles(self) -> List[RobotProfile]:
        action_capabilities: Dict[str, List[str]] = {
            directive.name: [] for directive in self.scenario.robots
        }
        available_tools: Dict[str, set[str]] = {
            directive.name: {DEFAULT_TOOL_NAME} for directive in self.scenario.robots
        }

        for step in self.scenario.plan:
            robot_name = step.get("robot")
            action_name = step.get("action")
            if not isinstance(robot_name, str) or not isinstance(action_name, str):
                continue

            action_key = canonical_action_name(action_name)
            if action_key not in action_capabilities[robot_name]:
                action_capabilities[robot_name].append(action_key)

            tool_name = step.get("tool")
            if isinstance(tool_name, str) and tool_name.strip():
                available_tools.setdefault(robot_name, {DEFAULT_TOOL_NAME}).add(tool_name.strip())

        profiles: List[RobotProfile] = []
        for priority, directive in enumerate(self.scenario.robots):
            capabilities = tuple(action_capabilities.get(directive.name, []))
            profiles.append(
                RobotProfile(
                    name=directive.name,
                    capabilities=capabilities,
                    start_location=directive.start_location,
                    available_tools=tuple(sorted(available_tools.get(directive.name, {DEFAULT_TOOL_NAME}))),
                    default_tool=DEFAULT_TOOL_NAME,
                    priority=priority,
                )
            )
        return profiles

    def build_behavior_tree(self) -> py_trees.trees.BehaviourTree:
        return build_multi_robot_tree_from_json(
            self.build_symbolic_plan(),
            robot_profiles=self.build_robot_profiles(),
        )

    def run(self, max_steps: int = 40) -> SimulationResult:
        history: List[StepSnapshot] = [
            StepSnapshot(
                tick=0,
                render=self.render(),
                events=("[GridWorld] Initial state",),
                frame=self.capture_frame(),
            )
        ]

        for tick in range(1, max_steps + 1):
            events = self.step()
            history.append(
                StepSnapshot(
                    tick=tick,
                    render=self.render(),
                    events=tuple(events),
                    frame=self.capture_frame(),
                )
            )
            if self.all_objects_delivered():
                return SimulationResult(
                    scenario=self.scenario,
                    steps_run=tick,
                    completed=True,
                    history=history,
                )

        return SimulationResult(
            scenario=self.scenario,
            steps_run=max_steps,
            completed=False,
            history=history,
        )

    def step(self) -> List[str]:
        events: List[str] = []
        occupied_positions = self._build_occupancy_counts()

        for directive in self.scenario.robots:
            robot = self.robots[directive.name]
            self._adjust_occupancy(occupied_positions, robot.position, -1)
            events.extend(
                self._execute_robot_step(
                    robot,
                    {position for position, count in occupied_positions.items() if count > 0},
                )
            )
            self._adjust_occupancy(occupied_positions, robot.position, 1)

        self._refresh_delivery_flags()
        self._advance_phase_if_complete()
        return events

    def render(self) -> str:
        target_positions = {
            self.resolve_named_position(condition.target)
            for condition in self.scenario.success_conditions
        }
        delivered_names = sorted(
            obj.name for obj in self.objects.values() if obj.delivered
        )

        rows: List[str] = []
        for y in range(self.height):
            row_chars: List[str] = []
            for x in range(self.width):
                position = (x, y)
                if position in self.walls:
                    row_chars.append("#")
                    continue

                robots_here = self._robots_at(position)
                object_here = self._object_at(position)
                is_target = position in target_positions

                if len(robots_here) > 1:
                    row_chars.append("*")
                    continue

                if robots_here:
                    row_chars.append(self.robot_display.get(robots_here[0].name, "r"))
                    continue

                if object_here is not None:
                    row_chars.append(_shape_symbol(object_here.shape))
                    continue

                if is_target:
                    row_chars.append("X")
                    continue

                row_chars.append(".")
            rows.append(" ".join(row_chars))

        footer = [
            "",
            "Legend: # wall, X goal/fixture, * stacked robots, o circle/gear, s square/shaft/plate, t triangle/pin, digits robots",
            "Delivered: {}".format(", ".join(delivered_names) if delivered_names else "none"),
            "Fixed: {}".format(
                ", ".join(
                    "{}@{}".format(name, position)
                    for name, position in sorted(
                        (
                            (location_name, location_position)
                            for location_name, location_position in self.named_locations.items()
                            if _fixed_location_category(location_name) is not None
                        ),
                        key=lambda item: item[0],
                    )
                )
            ),
            "Goals: {}".format(
                ", ".join(
                    "{}->{}@{}".format(
                        condition.object_name,
                        condition.target,
                        self.resolve_named_position(condition.target),
                    )
                    for condition in self.scenario.success_conditions
                )
            ),
        ]
        return "\n".join(rows + footer)

    def capture_frame(self) -> GridWorldFrame:
        return GridWorldFrame(
            width=self.width,
            height=self.height,
            walls=tuple(sorted(self.walls)),
            goals=tuple(
                (condition.object_name, condition.target, self.resolve_named_position(condition.target))
                for condition in self.scenario.success_conditions
            ),
            fixed_locations=tuple(
                GridWorldFrameLocation(
                    name=location_name,
                    category=location_category,
                    position=position,
                )
                for location_name, position in sorted(self.named_locations.items())
                for location_category in [_fixed_location_category(location_name)]
                if location_category is not None
            ),
            robots=tuple(
                GridWorldFrameRobot(
                    name=robot.name,
                    role=robot.role,
                    position=robot.position,
                    can_move=robot.can_move,
                    carrying=robot.carrying,
                    current_tool=robot.current_tool,
                )
                for robot in sorted(self.robots.values(), key=lambda item: item.name)
            ),
            objects=tuple(
                GridWorldFrameObject(
                    name=obj.name,
                    kind=obj.kind,
                    shape=obj.shape,
                    position=obj.position,
                    delivered=obj.delivered,
                    held_by=obj.held_by,
                    inserted_target=obj.inserted_target,
                )
                for obj in sorted(self.objects.values(), key=lambda item: item.name)
            ),
            current_phase_index=self.current_phase_index,
            total_phases=len(self.phase_plans),
        )

    def describe(self) -> str:
        object_summary = ", ".join(
            "{}({})@{}".format(obj.name, obj.kind, obj.position)
            for obj in self.objects.values()
        )
        robot_summary = ", ".join(
            "{}@{} role={} carrying={}".format(
                robot.name,
                robot.position,
                robot.role,
                robot.carrying or "nothing",
            )
            for robot in self.robots.values()
        )
        goal_summary = ", ".join(
            "{}->{}".format(condition.object_name, condition.target)
            for condition in self.scenario.success_conditions
        )
        return "layout={}, robots={}, objects={}, goals={}".format(
            self.layout.name,
            robot_summary,
            object_summary,
            goal_summary,
        )

    def resolve_named_position(self, location_name: str) -> Position:
        cleaned_name = location_name.strip()
        if cleaned_name not in self.named_locations:
            raise ValueError("Unsupported gridworld location '{}'.".format(cleaned_name))
        return self.named_locations[cleaned_name]

    def _spawn_robots(self) -> Dict[str, GridRobot]:
        robots: Dict[str, GridRobot] = {}
        for directive in self.scenario.robots:
            position = self.resolve_named_position(directive.start_location)
            robots[directive.name] = GridRobot(
                name=directive.name,
                role=directive.role,
                position=position,
                can_move=directive.can_move,
            )
        return robots

    def _spawn_objects(self) -> Dict[str, GridObject]:
        excluded_positions = {robot.position for robot in self.robots.values()}
        preferred_positions = self._pick_positions(
            self.layout.preferred_object_spawns,
            len(self.scenario.objects),
            exclude=excluded_positions,
        )

        objects: Dict[str, GridObject] = {}
        for directive, position in zip(self.scenario.objects, preferred_positions):
            objects[directive.name] = GridObject(
                name=directive.name,
                kind=directive.kind,
                shape=directive.shape,
                position=position,
            )
        return objects

    def _group_plan_by_phase(
        self,
        plan: Sequence[Dict[str, str]],
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        phase_plans: List[Dict[str, List[Dict[str, str]]]] = []
        for phase in group_segments_into_phases(segment_plan(plan)):
            grouped_phase = {directive.name: [] for directive in self.scenario.robots}
            for segment in phase:
                for step in segment.steps:
                    robot_name = step.get("robot")
                    if not isinstance(robot_name, str) or robot_name not in grouped_phase:
                        raise ValueError("Plan step must reference a known robot.")
                    grouped_phase[robot_name].append(dict(step))
            phase_plans.append(grouped_phase)

        if phase_plans:
            return phase_plans

        return [{directive.name: [] for directive in self.scenario.robots}]

    def _pick_positions(
        self,
        preferred_positions: Sequence[Position],
        count: int,
        exclude: Iterable[Position],
    ) -> List[Position]:
        chosen: List[Position] = []
        blocked = set(exclude)

        for position in preferred_positions:
            if len(chosen) >= count:
                break
            if self._can_place_at(position, blocked):
                chosen.append(position)
                blocked.add(position)

        if len(chosen) >= count:
            return chosen

        free_positions = [
            (x, y)
            for y in range(1, self.height - 1)
            for x in range(1, self.width - 1)
            if self._can_place_at((x, y), blocked)
        ]
        self.random.shuffle(free_positions)
        for position in free_positions:
            if len(chosen) >= count:
                break
            chosen.append(position)
            blocked.add(position)

        if len(chosen) < count:
            raise ValueError("Could not place all entities in the selected layout.")

        return chosen

    def _can_place_at(self, position: Position, blocked: Iterable[Position]) -> bool:
        return position not in self.walls and position not in blocked

    def _execute_robot_step(
        self,
        robot: GridRobot,
        occupied_positions: Iterable[Position],
    ) -> List[str]:
        current_step = self._current_step(robot)
        if current_step is None:
            support_location = self._resolve_handoff_support_location(robot)
            if support_location is not None:
                location_name = support_location[0]
                location_position = support_location[1]
                if robot.position == location_position:
                    return [
                        "[GridWorld] {} is waiting at {} to receive a handoff.".format(
                            robot.name,
                            location_name,
                        )
                    ]

                next_position = self._move_robot_towards(
                    robot,
                    goal=location_position,
                    occupied_positions=occupied_positions,
                )
                if next_position is not None:
                    return [
                        "[GridWorld] {} moved toward {} to receive a handoff.".format(
                            robot.name,
                            location_name,
                        )
                    ]
                return [
                    "[GridWorld] {} is blocked while moving toward {} to receive a handoff.".format(
                        robot.name,
                        location_name,
                    )
                ]

            return ["[GridWorld] {} is idle.".format(robot.name)]

        action_name = canonical_action_name(current_step.get("action", ""))
        object_name = current_step.get("object")
        target = current_step.get("target")
        location = current_step.get("location")
        tool = current_step.get("tool")
        recipient = current_step.get("recipient")

        if action_name == "Pick" and object_name:
            return [self._execute_pick(robot, object_name, occupied_positions)]

        if action_name == "MoveTo" and target:
            return [self._execute_move_to(robot, target, occupied_positions)]

        if action_name == "Place" and object_name and target:
            return [self._execute_place(robot, object_name, target, occupied_positions)]

        if action_name == "Insert" and object_name and target:
            return [self._execute_insert(robot, object_name, target, tool, occupied_positions)]

        if action_name == "ChangeTool" and tool:
            robot.current_tool = tool
            robot.next_step_index += 1
            return [
                "[GridWorld] {} switched to the {}.".format(
                    robot.name,
                    tool,
                )
            ]

        if action_name == "Handoff" and object_name and recipient and location:
            return [
                self._execute_handoff(
                    robot,
                    object_name,
                    recipient,
                    location,
                    occupied_positions,
                )
            ]

        raise ValueError(
            "Unsupported gridworld step '{}' for robot '{}'.".format(
                action_name,
                robot.name,
            )
        )

    def _execute_pick(
        self,
        robot: GridRobot,
        object_name: str,
        occupied_positions: Iterable[Position],
    ) -> str:
        obj = self.objects[object_name]

        if robot.carrying == object_name:
            robot.next_step_index += 1
            return "[GridWorld] {} is already holding {}.".format(robot.name, object_name)

        if obj.held_by is not None and obj.held_by != robot.name:
            return "[GridWorld] {} is waiting because {} is held by {}.".format(
                robot.name,
                object_name,
                obj.held_by,
            )

        if robot.position == obj.position:
            robot.carrying = object_name
            obj.held_by = robot.name
            robot.next_step_index += 1
            return "[GridWorld] {} picked up {}.".format(robot.name, object_name)

        next_position = self._move_robot_towards(
            robot,
            goal=obj.position,
            occupied_positions=occupied_positions,
        )
        if next_position is None:
            return "[GridWorld] {} is blocked while moving toward {}.".format(
                robot.name,
                object_name,
            )

        return "[GridWorld] {} moved toward {}.".format(robot.name, object_name)

    def _execute_move_to(
        self,
        robot: GridRobot,
        target_name: str,
        occupied_positions: Iterable[Position],
    ) -> str:
        target_position = self.resolve_named_position(target_name)
        if robot.position == target_position:
            robot.next_step_index += 1
            return "[GridWorld] {} is already at {}.".format(robot.name, target_name)

        next_position = self._move_robot_towards(
            robot,
            goal=target_position,
            occupied_positions=occupied_positions,
        )
        if next_position is None:
            return "[GridWorld] {} is blocked while moving toward {}.".format(
                robot.name,
                target_name,
            )

        if robot.position == target_position:
            robot.next_step_index += 1
            return "[GridWorld] {} reached {}.".format(robot.name, target_name)

        return "[GridWorld] {} moved toward {}.".format(robot.name, target_name)

    def _execute_place(
        self,
        robot: GridRobot,
        object_name: str,
        target_name: str,
        occupied_positions: Iterable[Position],
    ) -> str:
        obj = self.objects[object_name]
        target_position = self.resolve_named_position(target_name)

        if obj.held_by is None and obj.position == target_position and obj.inserted_target is None:
            robot.next_step_index += 1
            return "[GridWorld] {} confirmed {} at {}.".format(
                robot.name,
                object_name,
                target_name,
            )

        if robot.carrying != object_name:
            return "[GridWorld] {} is waiting to place {} because it is not holding it.".format(
                robot.name,
                object_name,
            )

        if robot.position == target_position:
            robot.carrying = None
            obj.held_by = None
            obj.inserted_target = None
            obj.position = target_position
            robot.next_step_index += 1
            return "[GridWorld] {} placed {} at {}.".format(
                robot.name,
                object_name,
                target_name,
            )

        next_position = self._move_robot_towards(
            robot,
            goal=target_position,
            occupied_positions=occupied_positions,
        )
        if next_position is None:
            return "[GridWorld] {} is blocked while carrying {} to {}.".format(
                robot.name,
                object_name,
                target_name,
            )

        return "[GridWorld] {} moved toward {} while carrying {}.".format(
            robot.name,
            target_name,
            object_name,
        )

    def _execute_insert(
        self,
        robot: GridRobot,
        object_name: str,
        target_name: str,
        required_tool: Optional[str],
        occupied_positions: Iterable[Position],
    ) -> str:
        obj = self.objects[object_name]
        target_position = self.resolve_named_position(target_name)

        if obj.inserted_target == target_name:
            robot.next_step_index += 1
            return "[GridWorld] {} confirmed {} inserted at {}.".format(
                robot.name,
                object_name,
                target_name,
            )

        allowed_targets = insert_targets_for_kind(obj.kind)
        if allowed_targets and target_name not in allowed_targets:
            return "[GridWorld] {} cannot insert {} into {} because {} fits {}.".format(
                robot.name,
                object_name,
                target_name,
                obj.kind,
                ", ".join(sorted(allowed_targets)),
            )

        if required_tool and robot.current_tool != required_tool:
            return "[GridWorld] {} is waiting to insert {} because {} is not equipped.".format(
                robot.name,
                object_name,
                required_tool,
            )

        if robot.carrying != object_name:
            return "[GridWorld] {} is waiting to insert {} because it is not holding it.".format(
                robot.name,
                object_name,
            )

        if robot.position == target_position:
            robot.carrying = None
            obj.held_by = None
            obj.position = target_position
            obj.inserted_target = target_name
            robot.next_step_index += 1
            return "[GridWorld] {} inserted {} into {}.".format(
                robot.name,
                object_name,
                target_name,
            )

        next_position = self._move_robot_towards(
            robot,
            goal=target_position,
            occupied_positions=occupied_positions,
        )
        if next_position is None:
            return "[GridWorld] {} is blocked while carrying {} to {}.".format(
                robot.name,
                object_name,
                target_name,
            )

        return "[GridWorld] {} moved toward {} while carrying {}.".format(
            robot.name,
            target_name,
            object_name,
        )

    def _execute_handoff(
        self,
        robot: GridRobot,
        object_name: str,
        recipient_name: str,
        location_name: str,
        occupied_positions: Iterable[Position],
    ) -> str:
        recipient = self.robots[recipient_name]
        location = self.resolve_named_position(location_name)

        if recipient.carrying == object_name and robot.carrying != object_name:
            robot.next_step_index += 1
            return "[GridWorld] {} confirmed {} is already with {}.".format(
                robot.name,
                object_name,
                recipient_name,
            )

        if robot.carrying != object_name:
            return "[GridWorld] {} is waiting to hand off {} because it is not holding it.".format(
                robot.name,
                object_name,
            )

        if robot.position != location:
            next_position = self._move_robot_towards(
                robot,
                goal=location,
                occupied_positions=occupied_positions,
            )
            if next_position is None:
                return "[GridWorld] {} is blocked while moving toward {} for handoff.".format(
                    robot.name,
                    location_name,
                )
            return "[GridWorld] {} moved toward {} while carrying {}.".format(
                robot.name,
                location_name,
                object_name,
            )

        if recipient.position != location:
            return "[GridWorld] {} is waiting for {} to arrive at {}.".format(
                robot.name,
                recipient_name,
                location_name,
            )

        if recipient.carrying is not None and recipient.carrying != object_name:
            return "[GridWorld] {} is waiting because {} is already holding {}.".format(
                robot.name,
                recipient_name,
                recipient.carrying,
            )

        robot.carrying = None
        recipient.carrying = object_name
        obj = self.objects[object_name]
        obj.held_by = recipient_name
        obj.position = recipient.position
        robot.next_step_index += 1
        return "[GridWorld] {} handed {} to {} at {}.".format(
            robot.name,
            object_name,
            recipient_name,
            location_name,
        )

    def _move_robot_towards(
        self,
        robot: GridRobot,
        goal: Position,
        occupied_positions: Iterable[Position],
    ) -> Optional[Position]:
        if robot.position == goal:
            return goal

        if not robot.can_move:
            return None

        next_position = self._next_step_towards(
            start=robot.position,
            goal=goal,
            occupied_positions=occupied_positions,
        )
        if next_position is None:
            return None

        robot.position = next_position
        if robot.carrying:
            carried_object = self.objects[robot.carrying]
            carried_object.position = next_position
            carried_object.held_by = robot.name
        return next_position

    def _current_step(self, robot: GridRobot) -> Optional[Dict[str, str]]:
        if self.current_phase_index >= len(self.phase_plans):
            return None

        steps = self.phase_plans[self.current_phase_index].get(robot.name, [])
        if robot.next_step_index >= len(steps):
            return None
        return steps[robot.next_step_index]

    def _resolve_handoff_support_location(
        self,
        robot: GridRobot,
    ) -> Optional[Tuple[str, Position]]:
        if self.current_phase_index >= len(self.phase_plans):
            return None

        current_phase = self.phase_plans[self.current_phase_index]
        for other_robot_name, steps in current_phase.items():
            if other_robot_name == robot.name:
                continue

            other_robot = self.robots.get(other_robot_name)
            if other_robot is None:
                continue

            step_index = other_robot.next_step_index
            if step_index >= len(steps):
                continue

            current_step = steps[step_index]
            action_name = canonical_action_name(current_step.get("action", ""))
            if action_name != "Handoff":
                continue
            if current_step.get("recipient") != robot.name:
                continue

            location_name = current_step.get("location")
            if not isinstance(location_name, str) or not location_name.strip():
                continue

            return location_name, self.resolve_named_position(location_name)

        return None

    def _next_step_towards(
        self,
        start: Position,
        goal: Position,
        occupied_positions: Iterable[Position],
    ) -> Optional[Position]:
        if start == goal:
            return start

        blocked_positions = {position for position in occupied_positions if position != goal}
        queue = deque([start])
        came_from: Dict[Position, Optional[Position]] = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                break

            for neighbor in self._neighbors(current):
                if neighbor in came_from:
                    continue
                if neighbor in self.walls:
                    continue
                if neighbor in blocked_positions:
                    continue
                came_from[neighbor] = current
                queue.append(neighbor)

        if goal not in came_from:
            return None

        step = goal
        while came_from[step] not in {None, start}:
            step = came_from[step]  # type: ignore[index]

        return step if came_from[step] == start else goal

    def _neighbors(self, position: Position) -> Iterable[Position]:
        x, y = position
        return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

    def _robots_at(self, position: Position) -> List[GridRobot]:
        return [robot for robot in self.robots.values() if robot.position == position]

    def _object_at(self, position: Position) -> Optional[GridObject]:
        for obj in self.objects.values():
            if obj.held_by is not None:
                continue
            if obj.position == position:
                return obj
        return None

    def _refresh_delivery_flags(self) -> None:
        for obj in self.objects.values():
            obj.delivered = any(
                condition.object_name == obj.name and self._is_success_condition_met(condition)
                for condition in self.scenario.success_conditions
            )

    def _is_success_condition_met(self, condition: GridWorldSuccessCondition) -> bool:
        obj = self.objects[condition.object_name]
        target_position = self.resolve_named_position(condition.target)
        if obj.inserted_target is not None:
            return obj.inserted_target == condition.target
        return obj.held_by is None and obj.position == target_position

    def _advance_phase_if_complete(self) -> None:
        while self.current_phase_index < len(self.phase_plans):
            current_phase = self.phase_plans[self.current_phase_index]
            if any(
                robot.next_step_index < len(current_phase.get(robot.name, []))
                for robot in self.robots.values()
            ):
                return

            self.current_phase_index += 1
            for robot in self.robots.values():
                robot.next_step_index = 0

    def _build_occupancy_counts(self) -> Dict[Position, int]:
        occupied_positions: Dict[Position, int] = {}
        for robot in self.robots.values():
            occupied_positions[robot.position] = occupied_positions.get(robot.position, 0) + 1
        return occupied_positions

    def _adjust_occupancy(
        self,
        occupied_positions: Dict[Position, int],
        position: Position,
        delta: int,
    ) -> None:
        new_count = occupied_positions.get(position, 0) + delta
        if new_count > 0:
            occupied_positions[position] = new_count
            return

        occupied_positions.pop(position, None)


def resolve_typed_scenario_text(explicit_text: Optional[str] = None) -> str:
    if explicit_text and explicit_text.strip():
        return explicit_text.strip()

    return DEFAULT_TYPED_SCENARIO


def prompt_for_typed_scenario() -> str:
    print("[GridWorld] Enter a typed gridworld scenario and press Enter.")
    print("[GridWorld] This demo now routes the scenario through the configured LLM planner.")
    print("[GridWorld] Leave it empty to use the default example:")
    print(DEFAULT_TYPED_SCENARIO)
    try:
        typed_text = input("\n[GridWorld] Scenario: ").strip()
    except EOFError:
        return DEFAULT_TYPED_SCENARIO

    return resolve_typed_scenario_text(typed_text)


def named_layout_locations(layout: GridLayout) -> Dict[str, Position]:
    """
    Provide a compact symbolic location vocabulary for the LLM.
    """

    width = layout.width
    height = layout.height
    preferred_locations = {
        "center": (width // 2, height // 2),
        "handoff_station": (width // 2, height // 2),
        "top_left": (1, 1),
        "top_right": (width - 2, 1),
        "bottom_left": (1, height - 2),
        "bottom_right": (width - 2, height - 2),
        "left_mid": (1, height // 2),
        "right_mid": (width - 2, height // 2),
        "upper_mid": (width // 2, 1),
        "lower_mid": (width // 2, height - 2),
        "top_center": (width // 2, 1),
        "bottom_center": (width // 2, height - 2),
        "left_center": (1, height // 2),
        "right_center": (width - 2, height // 2),
        "north_room": (width // 2, 2),
        "south_room": (width // 2, height - 3),
        "west_room": (2, height // 2),
        "east_room": (width - 3, height // 2),
        "tool_rack": (2, 1),
        "precision_station": (width // 2, 2),
        "chassis": (width - 3, height // 2),
        "bearing_block": (width // 2, height - 3),
        "panel_slot": (width - 3, 2),
        "assembly_station": (width // 2, height // 2),
    }

    return {
        name: _nearest_open_cell(layout, position)
        for name, position in preferred_locations.items()
    }


def build_scenario_from_payload(
    scenario_text: str,
    payload: Dict[str, Any],
    num_robots: int,
    allowed_locations: Sequence[str],
) -> TypedTransportScenario:
    if not isinstance(payload, dict):
        raise ValueError("Gridworld scenario payload must be a JSON object.")

    allowed_location_set = {location.strip() for location in allowed_locations if location.strip()}
    if not allowed_location_set:
        raise ValueError("Gridworld scenario requires at least one allowed location.")

    task_summary = payload.get("task_summary")
    if not isinstance(task_summary, str) or not task_summary.strip():
        task_summary = resolve_typed_scenario_text(scenario_text)

    raw_robots = payload.get("robots")
    if not isinstance(raw_robots, list) or len(raw_robots) != num_robots:
        raise ValueError(
            "Gridworld scenario must describe exactly {} robots.".format(num_robots)
        )

    robots: List[GridWorldRobotDirective] = []
    robot_names = set()
    for raw_robot in raw_robots:
        if not isinstance(raw_robot, dict):
            raise ValueError("Each gridworld robot spec must be a JSON object.")

        name = _required_string(raw_robot, "name")
        role = _required_string(raw_robot, "role")
        start_location = _required_string(raw_robot, "start_location")
        can_move = raw_robot.get("can_move")
        if not isinstance(can_move, bool):
            raise ValueError("Gridworld robot '{}' must define boolean can_move.".format(name))
        if start_location not in allowed_location_set:
            raise ValueError(
                "Gridworld robot '{}' uses unsupported start_location '{}'.".format(
                    name,
                    start_location,
                )
            )
        if name in robot_names:
            raise ValueError("Gridworld robot names must be unique.")

        robots.append(
            GridWorldRobotDirective(
                name=name,
                role=role,
                start_location=start_location,
                can_move=can_move,
            )
        )
        robot_names.add(name)
    robot_lookup = {robot.name: robot for robot in robots}

    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("Gridworld scenario must include a non-empty objects list.")

    objects: List[GridWorldObjectDirective] = []
    object_names = set()
    for raw_object in raw_objects:
        if not isinstance(raw_object, dict):
            raise ValueError("Each gridworld object spec must be a JSON object.")

        name = _required_string(raw_object, "name")
        raw_kind = raw_object.get("kind")
        kind = infer_object_kind(
            name=name,
            explicit_kind=raw_kind if isinstance(raw_kind, str) else None,
            shape=raw_object.get("shape") if isinstance(raw_object.get("shape"), str) else None,
        )
        if kind not in SUPPORTED_OBJECT_KINDS:
            raise ValueError(
                "Gridworld object '{}' must use a supported kind.".format(name)
            )

        raw_shape = raw_object.get("shape")
        shape = raw_shape.strip().lower() if isinstance(raw_shape, str) and raw_shape.strip() else default_render_shape(kind)
        if shape not in SUPPORTED_RENDER_SHAPES:
            raise ValueError(
                "Gridworld object '{}' must use a supported shape.".format(name)
            )
        if name in object_names:
            raise ValueError("Gridworld object names must be unique.")

        objects.append(GridWorldObjectDirective(name=name, kind=kind, shape=shape))
        object_names.add(name)
    object_lookup = {item.name: item for item in objects}

    raw_plan = payload.get("plan")
    if not isinstance(raw_plan, list) or not raw_plan:
        raise ValueError("Gridworld scenario must include a non-empty plan.")

    plan: List[Dict[str, str]] = []
    for raw_step in raw_plan:
        if not isinstance(raw_step, dict):
            raise ValueError("Each plan step must be a JSON object.")
        robot_name = _required_string(raw_step, "robot")
        action_name = canonical_action_name(_required_string(raw_step, "action"))
        if robot_name not in robot_names:
            raise ValueError("Plan step references unknown robot '{}'.".format(robot_name))

        normalized_step: Dict[str, str] = {"robot": robot_name, "action": action_name}
        for optional_field in ("object", "target", "tool", "recipient", "location"):
            value = raw_step.get(optional_field)
            if isinstance(value, str) and value.strip():
                normalized_step[optional_field] = value.strip()

        object_name = normalized_step.get("object")
        target = normalized_step.get("target")
        location = normalized_step.get("location")
        recipient = normalized_step.get("recipient")

        if object_name is not None and object_name not in object_names:
            raise ValueError("Plan step references unknown object '{}'.".format(object_name))
        if target is not None and target not in allowed_location_set:
            raise ValueError("Plan step uses unsupported target '{}'.".format(target))
        if location is not None and location not in allowed_location_set:
            raise ValueError("Plan step uses unsupported location '{}'.".format(location))
        if recipient is not None and recipient not in robot_names:
            raise ValueError("Plan step references unknown recipient '{}'.".format(recipient))

        robot_directive = robot_lookup[robot_name]
        if not robot_directive.can_move:
            if action_name == "MoveTo":
                raise ValueError(
                    "Stationary robot '{}' cannot use MoveTo.".format(robot_name)
                )
            required_location = location or target
            if required_location is not None and required_location != robot_directive.start_location:
                raise ValueError(
                    "Stationary robot '{}' must stay at '{}'.".format(
                        robot_name,
                        robot_directive.start_location,
                    )
                )
        if action_name == "Insert" and object_name is not None and target is not None:
            object_kind = object_lookup[object_name].kind
            if not is_insert_compatible(object_kind, target):
                allowed_targets = insert_targets_for_kind(object_kind)
                raise ValueError(
                    "Object '{}' of kind '{}' cannot be inserted into '{}'. Allowed targets: {}.".format(
                        object_name,
                        object_kind,
                        target,
                        ", ".join(sorted(allowed_targets)),
                    )
                )

        plan.append(normalized_step)

    raw_success_conditions = payload.get("success_conditions")
    if not isinstance(raw_success_conditions, list) or not raw_success_conditions:
        raise ValueError("Gridworld scenario must include success_conditions.")

    success_conditions: List[GridWorldSuccessCondition] = []
    for raw_condition in raw_success_conditions:
        if not isinstance(raw_condition, dict):
            raise ValueError("Each success condition must be a JSON object.")

        object_name = _required_string(raw_condition, "object")
        target = _required_string(raw_condition, "target")
        if object_name not in object_names:
            raise ValueError("Success condition references unknown object '{}'.".format(object_name))
        if target not in allowed_location_set:
            raise ValueError("Success condition uses unsupported target '{}'.".format(target))

        success_conditions.append(
            GridWorldSuccessCondition(
                object_name=object_name,
                target=target,
            )
        )

    return TypedTransportScenario(
        raw_text=resolve_typed_scenario_text(scenario_text),
        task_summary=task_summary.strip(),
        robots=tuple(robots),
        objects=tuple(objects),
        plan=tuple(plan),
        success_conditions=tuple(success_conditions),
    )


def build_env_from_typed_scenario(
    scenario_text: str,
    num_robots: int = 3,
    num_circles: Optional[int] = None,
    layout_name: str = "open_room",
    seed: int = 0,
    planner: Optional[LLMTaskPlanner] = None,
    scenario_payload: Optional[Dict[str, Any]] = None,
) -> TypedGridWorldEnv:
    layouts = layout_registry()
    if layout_name not in layouts:
        raise ValueError(
            "Unknown gridworld layout '{}'. Available layouts: {}.".format(
                layout_name,
                ", ".join(sorted(layouts)),
            )
        )

    layout = layouts[layout_name]
    allowed_locations = list(named_layout_locations(layout))
    resolved_text = resolve_typed_scenario_text(scenario_text)

    if scenario_payload is None:
        active_planner = planner or LLMTaskPlanner()
        scenario_payload = active_planner.plan_gridworld_task(
            instruction=resolved_text,
            num_robots=max(1, num_robots),
            num_circles=max(1, num_circles) if num_circles is not None else None,
            layout_name=layout_name,
            available_locations=allowed_locations,
        )

    scenario = build_scenario_from_payload(
        scenario_text=resolved_text,
        payload=scenario_payload,
        num_robots=max(1, num_robots),
        allowed_locations=allowed_locations,
    )
    return TypedGridWorldEnv(scenario=scenario, layout=layout, seed=seed)


def _shape_symbol(shape: str) -> str:
    if shape == "circle":
        return "o"
    if shape == "square":
        return "s"
    if shape == "triangle":
        return "t"
    return "?"


def _fixed_location_category(location_name: str) -> Optional[str]:
    if location_name in TOOL_STATION_LOCATIONS:
        return "tool_station"
    if location_name in FIXTURE_LOCATIONS:
        return "fixture"
    return None


def _nearest_open_cell(layout: GridLayout, position: Position) -> Position:
    if position not in layout.walls:
        return position

    queue = deque([position])
    visited = {position}
    while queue:
        current = queue.popleft()
        for neighbor in _neighbor_positions(current):
            x, y = neighbor
            if not (0 <= x < layout.width and 0 <= y < layout.height):
                continue
            if neighbor in visited:
                continue
            if neighbor not in layout.walls:
                return neighbor
            visited.add(neighbor)
            queue.append(neighbor)

    raise ValueError("Could not find an open cell near {}.".format(position))


def _neighbor_positions(position: Position) -> Iterable[Position]:
    x, y = position
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _required_string(payload: Dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing non-empty '{}'.".format(field_name))
    return value.strip()
