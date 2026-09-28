"""Orchestrator-Worker multi-agent backend.

Request flow:
    User -> Orchestrator -> Planner -> Worker Router -> Workers
         -> Evaluator -> Aggregator -> Final Answer
"""

__version__ = "0.1.0"