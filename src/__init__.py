"""
Core package for the LLM-as-BT planner prototype.

The package layout intentionally mirrors the planning pipeline:
natural language -> JSON task plan -> executable behavior tree.
"""

__all__ = [
    "bt_builder",
    "gridworld_app",
    "gridworld_env",
    "gridworld_layouts",
    "gridworld_presets",
    "llm_client",
    "multi_robot_actions",
    "multi_robot_planner",
    "robot_actions",
]

__version__ = "0.2.0"
