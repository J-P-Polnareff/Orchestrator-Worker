"""Aggregation layer: combine a finished execution into a final answer."""

from .base import Aggregator, AggregatorError
from .llm import LLMAggregator

__all__ = [
    "Aggregator",
    "AggregatorError",
    "LLMAggregator",
]