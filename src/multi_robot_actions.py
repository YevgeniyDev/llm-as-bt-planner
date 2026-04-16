"""
Shared state and runtime nodes for MRBTP-inspired multi-robot execution.

The existing project models a single robot with a tiny symbolic world. This
module keeps that spirit, but extends it to a small team that shares:

- object state
- per-robot location and gripper state
- lightweight intention broadcasts used to suppress redundant backup actions
"""

from dataclasses import dataclass, field
import re
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import py_trees


DEFAULT_OBJECT_LOCATION = "parts_bin"
DEFAULT_MULTI_ROBOT_CAPABILITIES = (
    "Pick",
    "MoveTo",
    "Place",
    "Insert",
    "ChangeTool",
    "Handoff",
)
LITERAL_PATTERN = re.compile(r"^\s*([A-Za-z_]+)\((.*?)\)\s*$")


def canonical_action_name(action_name: str) -> str:
    normalized = action_name.strip().replace("_", "").replace(" ", "").lower()
    mapping = {
        "pick": "Pick",
        "moveto": "MoveTo",
        "place": "Place",
        "insert": "Insert",
        "changetool": "ChangeTool",
        "handoff": "Handoff",
    }
    return mapping.get(normalized, action_name.strip())


def parse_literal(literal: str) -> Tuple[str, Tuple[str, ...]]:
    match = LITERAL_PATTERN.match(literal.strip())
    if not match:
        raise ValueError("Unsupported symbolic literal '{}'.".format(literal))

    predicate = match.group(1).strip()
    raw_args = match.group(2).strip()
    if not raw_args:
        return predicate, ()

    args = tuple(part.strip() for part in raw_args.split(",") if part.strip())
    return predicate, args


@dataclass(frozen=True)
class RobotProfile:
    name: str
    capabilities: Tuple[str, ...] = DEFAULT_MULTI_ROBOT_CAPABILITIES
    start_location: str = "home"
    available_tools: Tuple[str, ...] = ("default_gripper",)
    default_tool: str = "default_gripper"
    priority: int = 0

    @classmethod
    def from_payload(cls, payload: Dict[str, object], default_priority: int) -> "RobotProfile":
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("Each multi-robot profile must include a non-empty 'name'.")

        raw_capabilities = payload.get("capabilities")
        if isinstance(raw_capabilities, Sequence) and not isinstance(raw_capabilities, (str, bytes)):
            capabilities = tuple(
                canonical_action_name(str(capability))
                for capability in raw_capabilities
                if str(capability).strip()
            )
        else:
            capabilities = DEFAULT_MULTI_ROBOT_CAPABILITIES

        raw_tools = payload.get("available_tools")
        if isinstance(raw_tools, Sequence) and not isinstance(raw_tools, (str, bytes)):
            available_tools = tuple(str(tool).strip() for tool in raw_tools if str(tool).strip())
        else:
            available_tools = ("default_gripper",)

        start_location = str(payload.get("start_location", "home")).strip() or "home"
        default_tool = str(payload.get("default_tool", available_tools[0])).strip() or available_tools[0]
        priority = int(payload.get("priority", default_priority))
        return cls(
            name=name,
            capabilities=capabilities or DEFAULT_MULTI_ROBOT_CAPABILITIES,
            start_location=start_location,
            available_tools=available_tools or ("default_gripper",),
            default_tool=default_tool,
            priority=priority,
        )


@dataclass(frozen=True)
class ActionIntention:
    robot_name: str
    action_name: str
    predicted_add: FrozenSet[str] = frozenset()
    predicted_del: FrozenSet[str] = frozenset()
    priority: int = 0


@dataclass
class MultiRobotWorldState:
    robot_locations: Dict[str, str] = field(default_factory=dict)
    held_objects: Dict[str, Optional[str]] = field(default_factory=dict)
    equipped_tools: Dict[str, str] = field(default_factory=dict)
    available_tools: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    object_locations: Dict[str, str] = field(default_factory=dict)
    inserted_objects: Dict[str, str] = field(default_factory=dict)
    robot_priorities: Dict[str, int] = field(default_factory=dict)
    intentions: Dict[str, ActionIntention] = field(default_factory=dict)

    @classmethod
    def from_profiles_and_plan(
        cls,
        robot_profiles: Sequence[RobotProfile],
        plan: Sequence[Dict[str, str]],
    ) -> "MultiRobotWorldState":
        state = cls()
        for profile in robot_profiles:
            state.register_robot(profile)

        for step in plan:
            object_name = step.get("object") or step.get("item")
            if isinstance(object_name, str) and object_name.strip():
                state.register_object(object_name)

        return state

    def clone(self) -> "MultiRobotWorldState":
        return MultiRobotWorldState(
            robot_locations=dict(self.robot_locations),
            held_objects=dict(self.held_objects),
            equipped_tools=dict(self.equipped_tools),
            available_tools=dict(self.available_tools),
            object_locations=dict(self.object_locations),
            inserted_objects=dict(self.inserted_objects),
            robot_priorities=dict(self.robot_priorities),
            intentions={},
        )

    def register_robot(self, profile: RobotProfile) -> None:
        self.robot_locations.setdefault(profile.name, profile.start_location)
        self.held_objects.setdefault(profile.name, None)
        available_tools = tuple(profile.available_tools) or ("default_gripper",)
        default_tool = profile.default_tool if profile.default_tool in available_tools else available_tools[0]
        self.equipped_tools.setdefault(profile.name, default_tool)
        self.available_tools[profile.name] = available_tools
        self.robot_priorities[profile.name] = profile.priority

    def register_object(self, object_name: str) -> None:
        cleaned_name = object_name.strip()
        if not cleaned_name:
            return

        if cleaned_name in self.inserted_objects:
            return

        if self.find_robot_holding(cleaned_name) is not None:
            return

        self.object_locations.setdefault(cleaned_name, DEFAULT_OBJECT_LOCATION)

    def find_robot_holding(self, object_name: str) -> Optional[str]:
        cleaned_name = object_name.strip()
        for robot_name, held_object in self.held_objects.items():
            if held_object == cleaned_name:
                return robot_name
        return None

    def publish_intention(self, intention: ActionIntention) -> None:
        self.intentions[intention.robot_name] = intention

    def clear_intention(self, robot_name: str) -> None:
        self.intentions.pop(robot_name, None)

    def is_holding(self, robot_name: str, object_name: str) -> bool:
        return self.held_objects.get(robot_name) == object_name.strip()

    def is_at(self, robot_name: str, target: str) -> bool:
        return self.robot_locations.get(robot_name) == target.strip()

    def is_hand_empty(self, robot_name: str) -> bool:
        return self.held_objects.get(robot_name) is None

    def is_tool_equipped(self, robot_name: str, tool_name: str) -> bool:
        return self.equipped_tools.get(robot_name) == tool_name.strip()

    def is_object_at(self, object_name: str, target: str) -> bool:
        cleaned_name = object_name.strip()
        cleaned_target = target.strip()
        return (
            self.object_locations.get(cleaned_name) == cleaned_target
            or self.inserted_objects.get(cleaned_name) == cleaned_target
        )

    def is_inserted(self, object_name: str, target: str) -> bool:
        return self.inserted_objects.get(object_name.strip()) == target.strip()

    def literal_is_true(self, literal: str) -> bool:
        predicate, args = parse_literal(literal)

        if predicate == "Holding" and len(args) == 2:
            return self.is_holding(args[0], args[1])

        if predicate == "At" and len(args) == 2:
            return self.is_at(args[0], args[1])

        if predicate == "HandEmpty" and len(args) == 1:
            return self.is_hand_empty(args[0])

        if predicate == "Equipped" and len(args) == 2:
            return self.is_tool_equipped(args[0], args[1])

        if predicate == "ObjectAt" and len(args) == 2:
            return self.is_object_at(args[0], args[1])

        if predicate == "Inserted" and len(args) == 2:
            return self.is_inserted(args[0], args[1])

        raise ValueError("Unsupported symbolic literal '{}'.".format(literal))

    def literal_is_believed_true(self, literal: str, owner_robot: str) -> bool:
        owner_priority = self.robot_priorities.get(owner_robot, 10**6)
        relevant_intentions = sorted(
            (
                intention
                for intention in self.intentions.values()
                if intention.robot_name != owner_robot and intention.priority < owner_priority
            ),
            key=lambda item: item.priority,
        )

        for intention in relevant_intentions:
            if literal in intention.predicted_del:
                return False
            if literal in intention.predicted_add:
                return True

        return False

    def apply_predicted_effect(
        self,
        robot_name: str,
        action_name: str,
        object_name: Optional[str] = None,
        target: Optional[str] = None,
        tool: Optional[str] = None,
        recipient: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        action_key = canonical_action_name(action_name)
        cleaned_object = object_name.strip() if isinstance(object_name, str) else None
        cleaned_target = target.strip() if isinstance(target, str) else None
        cleaned_tool = tool.strip() if isinstance(tool, str) else None
        cleaned_recipient = recipient.strip() if isinstance(recipient, str) else None
        cleaned_location = location.strip() if isinstance(location, str) else None

        if cleaned_object:
            self.register_object(cleaned_object)

        if action_key == "Pick" and cleaned_object:
            self.inserted_objects.pop(cleaned_object, None)
            self.object_locations.pop(cleaned_object, None)
            self.held_objects[robot_name] = cleaned_object
            return

        if action_key == "MoveTo" and cleaned_target:
            self.robot_locations[robot_name] = cleaned_target
            return

        if action_key == "ChangeTool" and (cleaned_tool or cleaned_target):
            self.equipped_tools[robot_name] = cleaned_tool or cleaned_target
            return

        if action_key == "Place" and cleaned_object and cleaned_target:
            self.held_objects[robot_name] = None
            self.robot_locations[robot_name] = cleaned_target
            self.inserted_objects.pop(cleaned_object, None)
            self.object_locations[cleaned_object] = cleaned_target
            return

        if action_key == "Insert" and cleaned_object and cleaned_target:
            self.held_objects[robot_name] = None
            self.robot_locations[robot_name] = cleaned_target
            self.object_locations.pop(cleaned_object, None)
            self.inserted_objects[cleaned_object] = cleaned_target
            return

        if action_key == "Handoff" and cleaned_object and cleaned_recipient:
            self.held_objects[robot_name] = None
            self.held_objects[cleaned_recipient] = cleaned_object
            if cleaned_location:
                self.robot_locations[robot_name] = cleaned_location
                self.robot_locations[cleaned_recipient] = cleaned_location
            self.object_locations.pop(cleaned_object, None)
            self.inserted_objects.pop(cleaned_object, None)

    def to_symbolic_state(self) -> FrozenSet[str]:
        literals = []

        for robot_name, location in sorted(self.robot_locations.items()):
            literals.append("At({}, {})".format(robot_name, location))

        for robot_name, held_object in sorted(self.held_objects.items()):
            if held_object:
                literals.append("Holding({}, {})".format(robot_name, held_object))
            else:
                literals.append("HandEmpty({})".format(robot_name))

        for robot_name, tool_name in sorted(self.equipped_tools.items()):
            literals.append("Equipped({}, {})".format(robot_name, tool_name))

        for object_name, location in sorted(self.object_locations.items()):
            literals.append("ObjectAt({}, {})".format(object_name, location))

        for object_name, target in sorted(self.inserted_objects.items()):
            literals.append("Inserted({}, {})".format(object_name, target))

        return frozenset(literals)

    def summary(self) -> str:
        robot_summary = ", ".join(
            "{}@{} holding {} using {}".format(
                robot_name,
                self.robot_locations.get(robot_name, "unknown"),
                self.held_objects.get(robot_name) or "nothing",
                self.equipped_tools.get(robot_name, "unknown_tool"),
            )
            for robot_name in sorted(self.robot_locations)
        ) or "none"
        object_summary = ", ".join(
            "{}@{}".format(object_name, location)
            for object_name, location in sorted(self.object_locations.items())
        ) or "none"
        inserted_summary = ", ".join(
            "{}->{}".format(object_name, target)
            for object_name, target in sorted(self.inserted_objects.items())
        ) or "none"
        return "robots=[{}], placed=[{}], inserted=[{}]".format(
            robot_summary,
            object_summary,
            inserted_summary,
        )


class TeamLiteralCondition(py_trees.behaviour.Behaviour):
    """
    Generic symbolic condition for a multi-robot shared world.
    """

    def __init__(
        self,
        literal: str,
        world_state: MultiRobotWorldState,
        owner_robot: str,
    ) -> None:
        super().__init__(name=literal)
        self.literal = literal.strip()
        self.world_state = world_state
        self.owner_robot = owner_robot

    def update(self) -> py_trees.common.Status:
        if self.world_state.literal_is_true(self.literal):
            self.feedback_message = "{} already holds.".format(self.literal)
            return py_trees.common.Status.SUCCESS

        if self.world_state.literal_is_believed_true(self.literal, self.owner_robot):
            self.feedback_message = "{} is believed true from a higher-priority teammate.".format(
                self.literal
            )
            return py_trees.common.Status.SUCCESS

        self.feedback_message = "{} is not satisfied yet.".format(self.literal)
        return py_trees.common.Status.FAILURE


class TeamRobotAction(py_trees.behaviour.Behaviour):
    """
    Two-tick mock action that can broadcast symbolic intent.
    """

    def __init__(
        self,
        name: str,
        robot_name: str,
        start_message: str,
        success_message: str,
        world_state: MultiRobotWorldState,
        goal_literal: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.robot_name = robot_name
        self.start_message = start_message
        self.success_message = success_message
        self.world_state = world_state
        self.goal_literal = goal_literal.strip() if isinstance(goal_literal, str) else None
        self._started = False

    def initialise(self) -> None:
        self._started = False

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status != py_trees.common.Status.RUNNING:
            self.world_state.clear_intention(self.robot_name)

    def validate_preconditions(self) -> Optional[str]:
        return None

    def apply_effects(self) -> None:
        return None

    def predicted_add_literals(self) -> FrozenSet[str]:
        return frozenset()

    def predicted_del_literals(self) -> FrozenSet[str]:
        return frozenset()

    def build_intention(self) -> ActionIntention:
        predicted_add = set(self.predicted_add_literals())
        if self.goal_literal:
            predicted_add.add(self.goal_literal)

        return ActionIntention(
            robot_name=self.robot_name,
            action_name=self.name,
            predicted_add=frozenset(predicted_add),
            predicted_del=self.predicted_del_literals(),
            priority=self.world_state.robot_priorities.get(self.robot_name, 0),
        )

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            self.world_state.publish_intention(self.build_intention())
            self.feedback_message = self.start_message
            print(self.start_message)
            return py_trees.common.Status.RUNNING

        failure_message = self.validate_preconditions()
        if failure_message:
            self.world_state.clear_intention(self.robot_name)
            self.feedback_message = failure_message
            print(failure_message)
            return py_trees.common.Status.FAILURE

        self.apply_effects()
        self.world_state.clear_intention(self.robot_name)
        self.feedback_message = self.success_message
        print(self.success_message)
        return py_trees.common.Status.SUCCESS


class RobotPick(TeamRobotAction):
    def __init__(
        self,
        robot_name: str,
        object_name: str,
        world_state: MultiRobotWorldState,
        goal_literal: Optional[str] = None,
    ) -> None:
        self.object_name = object_name.strip()
        world_state.register_object(self.object_name)
        super().__init__(
            name="Pick({}, {})".format(robot_name, self.object_name),
            robot_name=robot_name,
            start_message="[{}] Reaching for the {}.".format(robot_name, self.object_name),
            success_message="[{}] Successfully picked up the {}.".format(
                robot_name, self.object_name
            ),
            world_state=world_state,
            goal_literal=goal_literal,
        )

    def validate_preconditions(self) -> Optional[str]:
        held_object = self.world_state.held_objects.get(self.robot_name)
        if held_object and held_object != self.object_name:
            return "[{}] Cannot pick {} because it is already holding {}.".format(
                self.robot_name,
                self.object_name,
                held_object,
            )

        current_holder = self.world_state.find_robot_holding(self.object_name)
        if current_holder and current_holder != self.robot_name:
            return "[{}] Cannot pick {} because {} already holds it.".format(
                self.robot_name,
                self.object_name,
                current_holder,
            )

        if self.object_name in self.world_state.inserted_objects:
            return "[{}] Cannot pick {} because it is already inserted.".format(
                self.robot_name,
                self.object_name,
            )

        return None

    def apply_effects(self) -> None:
        self.world_state.inserted_objects.pop(self.object_name, None)
        self.world_state.object_locations.pop(self.object_name, None)
        self.world_state.held_objects[self.robot_name] = self.object_name

    def predicted_add_literals(self) -> FrozenSet[str]:
        return frozenset({"Holding({}, {})".format(self.robot_name, self.object_name)})

    def predicted_del_literals(self) -> FrozenSet[str]:
        return frozenset({"HandEmpty({})".format(self.robot_name)})


class RobotMoveTo(TeamRobotAction):
    def __init__(
        self,
        robot_name: str,
        target: str,
        world_state: MultiRobotWorldState,
        goal_literal: Optional[str] = None,
    ) -> None:
        self.target = target.strip()
        super().__init__(
            name="MoveTo({}, {})".format(robot_name, self.target),
            robot_name=robot_name,
            start_message="[{}] Moving toward the {}.".format(robot_name, self.target),
            success_message="[{}] Successfully moved to the {}.".format(robot_name, self.target),
            world_state=world_state,
            goal_literal=goal_literal,
        )

    def apply_effects(self) -> None:
        self.world_state.robot_locations[self.robot_name] = self.target

    def predicted_add_literals(self) -> FrozenSet[str]:
        return frozenset({"At({}, {})".format(self.robot_name, self.target)})


class RobotPlace(TeamRobotAction):
    def __init__(
        self,
        robot_name: str,
        object_name: str,
        target: str,
        world_state: MultiRobotWorldState,
        goal_literal: Optional[str] = None,
    ) -> None:
        self.object_name = object_name.strip()
        self.target = target.strip()
        world_state.register_object(self.object_name)
        super().__init__(
            name="Place({}, {}, {})".format(robot_name, self.object_name, self.target),
            robot_name=robot_name,
            start_message="[{}] Positioning the {} over the {}.".format(
                robot_name,
                self.object_name,
                self.target,
            ),
            success_message="[{}] Successfully placed the {} on the {}.".format(
                robot_name,
                self.object_name,
                self.target,
            ),
            world_state=world_state,
            goal_literal=goal_literal,
        )

    def validate_preconditions(self) -> Optional[str]:
        if self.world_state.held_objects.get(self.robot_name) != self.object_name:
            return "[{}] Cannot place {} because it is not in the gripper.".format(
                self.robot_name,
                self.object_name,
            )

        if not self.world_state.is_at(self.robot_name, self.target):
            return "[{}] Cannot place {} because it is not at {}.".format(
                self.robot_name,
                self.object_name,
                self.target,
            )

        return None

    def apply_effects(self) -> None:
        self.world_state.held_objects[self.robot_name] = None
        self.world_state.object_locations[self.object_name] = self.target
        self.world_state.inserted_objects.pop(self.object_name, None)

    def predicted_add_literals(self) -> FrozenSet[str]:
        return frozenset({"ObjectAt({}, {})".format(self.object_name, self.target)})

    def predicted_del_literals(self) -> FrozenSet[str]:
        return frozenset({"Holding({}, {})".format(self.robot_name, self.object_name)})


class RobotInsert(TeamRobotAction):
    def __init__(
        self,
        robot_name: str,
        object_name: str,
        target: str,
        world_state: MultiRobotWorldState,
        required_tool: Optional[str] = None,
        goal_literal: Optional[str] = None,
    ) -> None:
        self.object_name = object_name.strip()
        self.target = target.strip()
        self.required_tool = required_tool.strip() if isinstance(required_tool, str) else None
        world_state.register_object(self.object_name)

        tool_suffix = ""
        if self.required_tool:
            tool_suffix = " using {}".format(self.required_tool)
        super().__init__(
            name="Insert({}, {}, {})".format(robot_name, self.object_name, self.target),
            robot_name=robot_name,
            start_message="[{}] Aligning the {} with the {}{}.".format(
                robot_name,
                self.object_name,
                self.target,
                tool_suffix,
            ),
            success_message="[{}] Successfully inserted the {} into the {}.".format(
                robot_name,
                self.object_name,
                self.target,
            ),
            world_state=world_state,
            goal_literal=goal_literal,
        )

    def validate_preconditions(self) -> Optional[str]:
        if self.world_state.held_objects.get(self.robot_name) != self.object_name:
            return "[{}] Cannot insert {} because it is not in the gripper.".format(
                self.robot_name,
                self.object_name,
            )

        if not self.world_state.is_at(self.robot_name, self.target):
            return "[{}] Cannot insert {} because it is not at {}.".format(
                self.robot_name,
                self.object_name,
                self.target,
            )

        if self.required_tool and not self.world_state.is_tool_equipped(
            self.robot_name, self.required_tool
        ):
            return "[{}] Cannot insert {} because {} is not equipped.".format(
                self.robot_name,
                self.object_name,
                self.required_tool,
            )

        return None

    def apply_effects(self) -> None:
        self.world_state.held_objects[self.robot_name] = None
        self.world_state.inserted_objects[self.object_name] = self.target
        self.world_state.object_locations.pop(self.object_name, None)

    def predicted_add_literals(self) -> FrozenSet[str]:
        return frozenset({"Inserted({}, {})".format(self.object_name, self.target)})

    def predicted_del_literals(self) -> FrozenSet[str]:
        return frozenset({"Holding({}, {})".format(self.robot_name, self.object_name)})


class RobotChangeTool(TeamRobotAction):
    def __init__(
        self,
        robot_name: str,
        tool_name: str,
        world_state: MultiRobotWorldState,
        goal_literal: Optional[str] = None,
    ) -> None:
        self.tool_name = tool_name.strip()
        super().__init__(
            name="ChangeTool({}, {})".format(robot_name, self.tool_name),
            robot_name=robot_name,
            start_message="[{}] Switching to the {}.".format(robot_name, self.tool_name),
            success_message="[{}] Successfully equipped the {}.".format(
                robot_name,
                self.tool_name,
            ),
            world_state=world_state,
            goal_literal=goal_literal,
        )

    def validate_preconditions(self) -> Optional[str]:
        available_tools = self.world_state.available_tools.get(self.robot_name, ())
        if self.tool_name not in available_tools:
            return "[{}] Cannot switch to {} because it is not available.".format(
                self.robot_name,
                self.tool_name,
            )

        return None

    def apply_effects(self) -> None:
        self.world_state.equipped_tools[self.robot_name] = self.tool_name

    def predicted_add_literals(self) -> FrozenSet[str]:
        return frozenset({"Equipped({}, {})".format(self.robot_name, self.tool_name)})

    def predicted_del_literals(self) -> FrozenSet[str]:
        available_tools = self.world_state.available_tools.get(self.robot_name, ())
        return frozenset(
            "Equipped({}, {})".format(self.robot_name, tool_name)
            for tool_name in available_tools
            if tool_name != self.tool_name
        )


class RobotHandoff(TeamRobotAction):
    def __init__(
        self,
        robot_name: str,
        object_name: str,
        recipient: str,
        location: str,
        world_state: MultiRobotWorldState,
        goal_literal: Optional[str] = None,
    ) -> None:
        self.object_name = object_name.strip()
        self.recipient = recipient.strip()
        self.location = location.strip()
        world_state.register_object(self.object_name)
        super().__init__(
            name="Handoff({}, {}, {}, {})".format(
                robot_name,
                self.object_name,
                self.recipient,
                self.location,
            ),
            robot_name=robot_name,
            start_message="[{}] Preparing to hand {} to {} at the {}.".format(
                robot_name,
                self.object_name,
                self.recipient,
                self.location,
            ),
            success_message="[{}] Successfully handed {} to {} at the {}.".format(
                robot_name,
                self.object_name,
                self.recipient,
                self.location,
            ),
            world_state=world_state,
            goal_literal=goal_literal,
        )

    def validate_preconditions(self) -> Optional[str]:
        if self.robot_name == self.recipient:
            return "[{}] Cannot hand {} to itself.".format(self.robot_name, self.object_name)

        if self.world_state.held_objects.get(self.robot_name) != self.object_name:
            return "[{}] Cannot hand off {} because it is not in the gripper.".format(
                self.robot_name,
                self.object_name,
            )

        if not self.world_state.is_at(self.robot_name, self.location):
            return "[{}] Cannot hand off {} because it is not at {}.".format(
                self.robot_name,
                self.object_name,
                self.location,
            )

        if not self.world_state.is_at(self.recipient, self.location):
            return "[{}] Cannot receive {} because it is not at {}.".format(
                self.recipient,
                self.object_name,
                self.location,
            )

        recipient_holding = self.world_state.held_objects.get(self.recipient)
        if recipient_holding is not None:
            return "[{}] Cannot receive {} because it is already holding {}.".format(
                self.recipient,
                self.object_name,
                recipient_holding,
            )

        return None

    def apply_effects(self) -> None:
        self.world_state.held_objects[self.robot_name] = None
        self.world_state.held_objects[self.recipient] = self.object_name
        self.world_state.robot_locations[self.robot_name] = self.location
        self.world_state.robot_locations[self.recipient] = self.location

    def predicted_add_literals(self) -> FrozenSet[str]:
        return frozenset(
            {
                "Holding({}, {})".format(self.recipient, self.object_name),
                "HandEmpty({})".format(self.robot_name),
            }
        )

    def predicted_del_literals(self) -> FrozenSet[str]:
        return frozenset(
            {
                "Holding({}, {})".format(self.robot_name, self.object_name),
                "HandEmpty({})".format(self.recipient),
            }
        )


def build_action_node(
    robot_name: str,
    action_name: str,
    world_state: MultiRobotWorldState,
    object_name: Optional[str] = None,
    target: Optional[str] = None,
    tool: Optional[str] = None,
    recipient: Optional[str] = None,
    location: Optional[str] = None,
    goal_literal: Optional[str] = None,
) -> TeamRobotAction:
    action_key = canonical_action_name(action_name)

    if action_key == "Pick" and object_name:
        return RobotPick(robot_name, object_name, world_state, goal_literal=goal_literal)

    if action_key == "MoveTo" and target:
        return RobotMoveTo(robot_name, target, world_state, goal_literal=goal_literal)

    if action_key == "ChangeTool" and (tool or target):
        return RobotChangeTool(
            robot_name,
            tool or target or "",
            world_state,
            goal_literal=goal_literal,
        )

    if action_key == "Place" and object_name and target:
        return RobotPlace(
            robot_name,
            object_name,
            target,
            world_state,
            goal_literal=goal_literal,
        )

    if action_key == "Insert" and object_name and target:
        return RobotInsert(
            robot_name,
            object_name,
            target,
            world_state,
            required_tool=tool,
            goal_literal=goal_literal,
        )

    if action_key == "Handoff" and object_name and recipient and location:
        return RobotHandoff(
            robot_name,
            object_name,
            recipient,
            location,
            world_state,
            goal_literal=goal_literal,
        )

    raise ValueError(
        "Unsupported multi-robot action '{}({}, {}, {}, {}, {})'.".format(
            action_name,
            robot_name,
            object_name or "",
            target or "",
            recipient or "",
            location or "",
        )
    )
