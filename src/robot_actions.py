"""
Mock robot behaviours used by the behavior tree executor.

These classes intentionally simulate robot actions instead of integrating with a
real middleware stack such as ROS. That keeps the repository lightweight while
still demonstrating the most important research idea from the paper:
an LLM-generated symbolic plan can be turned into executable behavior tree nodes.
"""

from typing import Optional

import py_trees


class MockRobotAction(py_trees.behaviour.Behaviour):
    """
    Base class for simple, deterministic robot action mocks.

    Each action takes two ticks:
    1. First tick: report that execution has started and return RUNNING.
    2. Second tick: report success and return SUCCESS.

    This small delay is deliberate. It makes the behavior tree's ticking
    semantics visible during a demo, and it also highlights why the root
    sequence uses memory=True in the builder: previously completed children
    should not be re-executed on every tick.
    """

    def __init__(self, name: str, start_message: str, success_message: str) -> None:
        super().__init__(name=name)
        self.start_message = start_message
        self.success_message = success_message
        self._started = False

    def initialise(self) -> None:
        """
        Reset transient execution state whenever the tree enters this node fresh.

        `py_trees` calls initialise() when the behaviour transitions from a
        non-running state into RUNNING. Resetting here makes the node reusable if
        the tree is executed again in a later experiment.
        """

        self._started = False

    def update(self) -> py_trees.common.Status:
        """
        Simulate a short-running robot action with clear console logging.
        """

        if not self._started:
            self._started = True
            self.feedback_message = self.start_message
            print(self.start_message)
            return py_trees.common.Status.RUNNING

        self.feedback_message = self.success_message
        print(self.success_message)
        return py_trees.common.Status.SUCCESS


class Pick(MockRobotAction):
    """Simulated grasp action."""

    def __init__(self, object_name: str) -> None:
        cleaned_name = object_name.strip()
        super().__init__(
            name="Pick({})".format(cleaned_name),
            start_message="[Robot] Reaching for the {}.".format(cleaned_name),
            success_message="[Robot] Successfully picked up the {}.".format(cleaned_name),
        )


class Place(MockRobotAction):
    """Simulated placement action."""

    def __init__(self, object_name: str, target: str) -> None:
        cleaned_name = object_name.strip()
        cleaned_target = target.strip()
        super().__init__(
            name="Place({}, {})".format(cleaned_name, cleaned_target),
            start_message="[Robot] Positioning the {} over the {}.".format(
                cleaned_name, cleaned_target
            ),
            success_message="[Robot] Successfully placed the {} on the {}.".format(
                cleaned_name, cleaned_target
            ),
        )


class MoveTo(MockRobotAction):
    """Simulated navigation or arm motion action."""

    def __init__(self, target: str) -> None:
        cleaned_target = target.strip()
        super().__init__(
            name="MoveTo({})".format(cleaned_target),
            start_message="[Robot] Moving toward the {}.".format(cleaned_target),
            success_message="[Robot] Successfully moved to the {}.".format(cleaned_target),
        )


class Insert(MockRobotAction):
    """Simulated insertion or assembly action."""

    def __init__(self, object_name: str, target: str, alignment_hint: Optional[str] = None) -> None:
        cleaned_name = object_name.strip()
        cleaned_target = target.strip()

        # The optional hint leaves room for future research extensions such as
        # skill-parameter grounding or perception-informed insertion policies.
        hint_suffix = ""
        if alignment_hint:
            hint_suffix = " using {}".format(alignment_hint.strip())

        super().__init__(
            name="Insert({}, {})".format(cleaned_name, cleaned_target),
            start_message="[Robot] Aligning the {} with the {}{}.".format(
                cleaned_name, cleaned_target, hint_suffix
            ),
            success_message="[Robot] Successfully inserted the {} into the {}.".format(
                cleaned_name, cleaned_target
            ),
        )
