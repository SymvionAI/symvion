"""
LegacyAgent doctrine layer: middleware for doctrine-aligned outputs.
"""

from app.runtime.agents.legacy_agent.doctrine_models import (
    DoctrineEvaluationResult,
    FilterScore,
    DOCTRINE_FILTERS,
    empty_evaluation,
)
from app.runtime.agents.legacy_agent.doctrine_cache import get_principles, clear_cache
from app.runtime.agents.legacy_agent.doctrine_engine import evaluate as doctrine_engine_evaluate
from app.runtime.agents.legacy_agent.doctrine_router import route as doctrine_router_route
from app.runtime.agents.legacy_agent.legacy_agent import LegacyAgent

__all__ = [
    "DoctrineEvaluationResult",
    "FilterScore",
    "DOCTRINE_FILTERS",
    "empty_evaluation",
    "get_principles",
    "clear_cache",
    "doctrine_engine_evaluate",
    "doctrine_router_route",
    "LegacyAgent",
]
