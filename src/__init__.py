"""
Core package for the LLM-as-BT planner prototype.

The package layout intentionally mirrors the planning pipeline:
natural language -> JSON task plan -> executable behavior tree.
"""

__all__ = [
    "bt_builder",
    "llm_client",
    "robot_actions",
]

__version__ = "0.1.0"
