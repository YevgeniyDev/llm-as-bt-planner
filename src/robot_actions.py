"""
Mock robot behaviours and state checks used by the behavior tree executor.

The original prototype only modelled imperative action nodes. This version adds
explicit condition nodes and a lightweight world state so the compiler can build
reactive subtrees such as:

Fallback(Holding(gear), Pick(gear))

That mirrors the paper's emphasis on behavior trees as state-aware policies
rather than flat action lists.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import py_trees


DEFAULT_OBJECT_LOCATION = "parts_bin"


@dataclass
class RobotWorldState:
    """
    Tiny in-memory simulation of robot and object state for reactive BT checks.
    """

    robot_location: str = "home"
    held_object: Optional[str] = None
    object_locations: Dict[str, str] = field(default_factory=dict)
    inserted_objects: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_plan(cls, plan: List[Dict[str, str]]) -> "RobotWorldState":
        """
        Seed a default scene from the objects mentioned in the symbolic plan.
        """

        state = cls()
        for step in plan:
            object_name = step.get("object") or step.get("item")
            if isinstance(object_name, str) and object_name.strip():
                state.register_object(object_name)
        return state

    def register_object(self, object_name: str) -> None:
        """
        Ensure a referenced object exists somewhere in the mock workspace.
        """

        cleaned_name = object_name.strip()
        if not cleaned_name:
            return

        if cleaned_name == self.held_object:
            return

        if cleaned_name in self.inserted_objects:
            return

        self.object_locations.setdefault(cleaned_name, DEFAULT_OBJECT_LOCATION)

    def is_holding(self, object_name: str) -> bool:
        return self.held_object == object_name.strip()

    def is_at(self, target: str) -> bool:
        return self.robot_location == target.strip()

    def is_object_at(self, object_name: str, target: str) -> bool:
        cleaned_name = object_name.strip()
        cleaned_target = target.strip()
        return (
            self.object_locations.get(cleaned_name) == cleaned_target
            or self.inserted_objects.get(cleaned_name) == cleaned_target
        )

    def is_inserted(self, object_name: str, target: str) -> bool:
        return self.inserted_objects.get(object_name.strip()) == target.strip()

    def summary(self) -> str:
        """
        Produce a compact final-state summary for demos.
        """

        held = self.held_object or "nothing"
        placed_items = (
            ", ".join(
                "{}@{}".format(name, location)
                for name, location in sorted(self.object_locations.items())
            )
            or "none"
        )
        inserted_items = (
            ", ".join(
                "{}->{}".format(name, location)
                for name, location in sorted(self.inserted_objects.items())
            )
            or "none"
        )
        return (
            "robot_at={}, holding={}, placed={}, inserted={}".format(
                self.robot_location, held, placed_items, inserted_items
            )
        )

    def clone(self) -> "RobotWorldState":
        """
        Create a deep-enough copy for recursive predicted-state rollouts.
        """

        return RobotWorldState(
            robot_location=self.robot_location,
            held_object=self.held_object,
            object_locations=dict(self.object_locations),
            inserted_objects=dict(self.inserted_objects),
        )

    def drop_held_object(self, location: Optional[str] = None) -> Optional[str]:
        """
        Simulate an accidental drop so reactive recovery can be demonstrated.
        """

        if not self.held_object:
            return None

        dropped_object = self.held_object
        drop_location = (
            location.strip()
            if isinstance(location, str) and location.strip()
            else self.robot_location
        )

        self.held_object = None
        self.inserted_objects.pop(dropped_object, None)
        self.object_locations[dropped_object] = drop_location
        return dropped_object

    def apply_symbolic_action(self, step: Dict[str, Any]) -> None:
        """
        Predict the state transition for a symbolic action step.

        This is intentionally deterministic and optimistic: it mirrors the
        intended effects of a valid plan so recursive planning can reason about
        future state without needing to execute the real BT first.
        """

        action_name = step.get("action")
        if not isinstance(action_name, str):
            return

        action_key = action_name.strip().replace("_", "").replace(" ", "").lower()
        object_name = step.get("object") or step.get("item")
        target = step.get("target") or step.get("destination") or step.get("location")

        cleaned_object_name = object_name.strip() if isinstance(object_name, str) else None
        cleaned_target = target.strip() if isinstance(target, str) else None

        if cleaned_object_name:
            self.register_object(cleaned_object_name)

        if action_key == "pick" and cleaned_object_name:
            self.inserted_objects.pop(cleaned_object_name, None)
            self.object_locations.pop(cleaned_object_name, None)
            self.held_object = cleaned_object_name
            return

        if action_key == "moveto" and cleaned_target:
            self.robot_location = cleaned_target
            return

        if action_key == "place" and cleaned_object_name and cleaned_target:
            self.held_object = None
            self.inserted_objects.pop(cleaned_object_name, None)
            self.object_locations[cleaned_object_name] = cleaned_target
            self.robot_location = cleaned_target
            return

        if action_key == "insert" and cleaned_object_name and cleaned_target:
            self.held_object = None
            self.inserted_objects[cleaned_object_name] = cleaned_target
            self.object_locations.pop(cleaned_object_name, None)
            self.robot_location = cleaned_target

    def apply_plan(self, plan: List[Dict[str, Any]]) -> "RobotWorldState":
        """
        Predict the final state after a symbolic plan.
        """

        for step in plan:
            self.apply_symbolic_action(step)
        return self


class WorldStateCondition(py_trees.behaviour.Behaviour):
    """
    Base class for condition checks that guard reactive subtrees.
    """

    def __init__(
        self,
        name: str,
        success_message: str,
        failure_message: str,
    ) -> None:
        super().__init__(name=name)
        self._success_message = success_message
        self._failure_message = failure_message

    def evaluate(self) -> bool:
        raise NotImplementedError

    def update(self) -> py_trees.common.Status:
        if self.evaluate():
            self.feedback_message = self._success_message
            return py_trees.common.Status.SUCCESS

        self.feedback_message = self._failure_message
        return py_trees.common.Status.FAILURE


class Holding(WorldStateCondition):
    """Check whether the gripper already holds the requested object."""

    def __init__(self, object_name: str, world_state: RobotWorldState) -> None:
        self.object_name = object_name.strip()
        self.world_state = world_state
        self.world_state.register_object(self.object_name)
        super().__init__(
            name="Holding({})".format(self.object_name),
            success_message="Already holding {}.".format(self.object_name),
            failure_message="Not holding {} yet.".format(self.object_name),
        )

    def evaluate(self) -> bool:
        return self.world_state.is_holding(self.object_name)


class AtLocation(WorldStateCondition):
    """Check whether the robot is already at the requested location."""

    def __init__(self, target: str, world_state: RobotWorldState) -> None:
        self.target = target.strip()
        self.world_state = world_state
        super().__init__(
            name="At({})".format(self.target),
            success_message="Already at {}.".format(self.target),
            failure_message="Not at {} yet.".format(self.target),
        )

    def evaluate(self) -> bool:
        return self.world_state.is_at(self.target)


class ObjectAt(WorldStateCondition):
    """Check whether an object is already placed at the target location."""

    def __init__(self, object_name: str, target: str, world_state: RobotWorldState) -> None:
        self.object_name = object_name.strip()
        self.target = target.strip()
        self.world_state = world_state
        self.world_state.register_object(self.object_name)
        super().__init__(
            name="ObjectAt({}, {})".format(self.object_name, self.target),
            success_message="{} is already at {}.".format(self.object_name, self.target),
            failure_message="{} is not at {} yet.".format(self.object_name, self.target),
        )

    def evaluate(self) -> bool:
        return self.world_state.is_object_at(self.object_name, self.target)


class InsertedAt(WorldStateCondition):
    """Check whether an object has already been inserted into its target."""

    def __init__(self, object_name: str, target: str, world_state: RobotWorldState) -> None:
        self.object_name = object_name.strip()
        self.target = target.strip()
        self.world_state = world_state
        self.world_state.register_object(self.object_name)
        super().__init__(
            name="Inserted({}, {})".format(self.object_name, self.target),
            success_message="{} is already inserted into {}.".format(
                self.object_name, self.target
            ),
            failure_message="{} is not inserted into {} yet.".format(
                self.object_name, self.target
            ),
        )

    def evaluate(self) -> bool:
        return self.world_state.is_inserted(self.object_name, self.target)


class MockRobotAction(py_trees.behaviour.Behaviour):
    """
    Base class for deterministic robot action mocks with state effects.

    Each action still takes two ticks so the demo remains easy to follow:
    1. First tick: announce start and return RUNNING.
    2. Second tick: validate preconditions, apply effects, and return SUCCESS.
    """

    def __init__(
        self,
        name: str,
        start_message: str,
        success_message: str,
        world_state: RobotWorldState,
    ) -> None:
        super().__init__(name=name)
        self.start_message = start_message
        self.success_message = success_message
        self.world_state = world_state
        self._started = False

    def initialise(self) -> None:
        self._started = False

    def validate_preconditions(self) -> Optional[str]:
        return None

    def apply_effects(self) -> None:
        return None

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            self.feedback_message = self.start_message
            print(self.start_message)
            return py_trees.common.Status.RUNNING

        failure_message = self.validate_preconditions()
        if failure_message:
            self.feedback_message = failure_message
            print(failure_message)
            return py_trees.common.Status.FAILURE

        self.apply_effects()
        self.feedback_message = self.success_message
        print(self.success_message)
        return py_trees.common.Status.SUCCESS


class Pick(MockRobotAction):
    """Simulated grasp action."""

    def __init__(self, object_name: str, world_state: RobotWorldState) -> None:
        self.object_name = object_name.strip()
        world_state.register_object(self.object_name)
        super().__init__(
            name="Pick({})".format(self.object_name),
            start_message="[Robot] Reaching for the {}.".format(self.object_name),
            success_message="[Robot] Successfully picked up the {}.".format(self.object_name),
            world_state=world_state,
        )

    def validate_preconditions(self) -> Optional[str]:
        if self.world_state.held_object and self.world_state.held_object != self.object_name:
            return (
                "[Robot] Cannot pick the {} because the gripper is already holding the {}.".format(
                    self.object_name, self.world_state.held_object
                )
            )

        if self.object_name in self.world_state.inserted_objects:
            return "[Robot] Cannot pick the {} because it is already inserted.".format(
                self.object_name
            )

        return None

    def apply_effects(self) -> None:
        self.world_state.inserted_objects.pop(self.object_name, None)
        self.world_state.object_locations.pop(self.object_name, None)
        self.world_state.held_object = self.object_name


class Place(MockRobotAction):
    """Simulated placement action."""

    def __init__(self, object_name: str, target: str, world_state: RobotWorldState) -> None:
        self.object_name = object_name.strip()
        self.target = target.strip()
        world_state.register_object(self.object_name)
        super().__init__(
            name="Place({}, {})".format(self.object_name, self.target),
            start_message="[Robot] Positioning the {} over the {}.".format(
                self.object_name, self.target
            ),
            success_message="[Robot] Successfully placed the {} on the {}.".format(
                self.object_name, self.target
            ),
            world_state=world_state,
        )

    def validate_preconditions(self) -> Optional[str]:
        if self.world_state.held_object != self.object_name:
            return "[Robot] Cannot place the {} because it is not in the gripper.".format(
                self.object_name
            )

        if not self.world_state.is_at(self.target):
            return "[Robot] Cannot place the {} because the robot is not at the {}.".format(
                self.object_name, self.target
            )

        return None

    def apply_effects(self) -> None:
        self.world_state.held_object = None
        self.world_state.inserted_objects.pop(self.object_name, None)
        self.world_state.object_locations[self.object_name] = self.target


class MoveTo(MockRobotAction):
    """Simulated navigation or arm motion action."""

    def __init__(self, target: str, world_state: RobotWorldState) -> None:
        self.target = target.strip()
        super().__init__(
            name="MoveTo({})".format(self.target),
            start_message="[Robot] Moving toward the {}.".format(self.target),
            success_message="[Robot] Successfully moved to the {}.".format(self.target),
            world_state=world_state,
        )

    def apply_effects(self) -> None:
        self.world_state.robot_location = self.target


class Insert(MockRobotAction):
    """Simulated insertion or assembly action."""

    def __init__(
        self,
        object_name: str,
        target: str,
        world_state: RobotWorldState,
        alignment_hint: Optional[str] = None,
    ) -> None:
        self.object_name = object_name.strip()
        self.target = target.strip()
        world_state.register_object(self.object_name)

        hint_suffix = ""
        if alignment_hint:
            hint_suffix = " using {}".format(alignment_hint.strip())

        super().__init__(
            name="Insert({}, {})".format(self.object_name, self.target),
            start_message="[Robot] Aligning the {} with the {}{}.".format(
                self.object_name, self.target, hint_suffix
            ),
            success_message="[Robot] Successfully inserted the {} into the {}.".format(
                self.object_name, self.target
            ),
            world_state=world_state,
        )

    def validate_preconditions(self) -> Optional[str]:
        if self.world_state.held_object != self.object_name:
            return "[Robot] Cannot insert the {} because it is not in the gripper.".format(
                self.object_name
            )

        if not self.world_state.is_at(self.target):
            return "[Robot] Cannot insert the {} because the robot is not at the {}.".format(
                self.object_name, self.target
            )

        return None

    def apply_effects(self) -> None:
        self.world_state.held_object = None
        self.world_state.inserted_objects[self.object_name] = self.target
        self.world_state.object_locations.pop(self.object_name, None)
