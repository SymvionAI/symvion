"""
Structured models for doctrine evaluation results.
Used by both DoctrineEngine (fast path) and LegacyAgent (strategic path).
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


# Canonical filter names for the eight doctrine dimensions
DOCTRINE_FILTERS = [
    "institutional_strength",
    "governance_integrity",
    "execution_readiness",
    "strategic_leverage",
    "talent_impact",
    "capital_logic",
    "ecosystem_relevance",
    "vision_alignment",
]


class FilterScore(BaseModel):
    """Score, reasoning, and evidence for a single doctrine filter."""

    score: int = Field(..., ge=0, le=100, description="Alignment score 0-100")
    reasoning: str = Field(default="", description="Brief reasoning for the score")
    evidence: List[str] = Field(default_factory=list, description="Evidence references from input")


class DoctrineEvaluationResult(BaseModel):
    """Structured output from doctrine evaluation (engine or LegacyAgent)."""

    institutional_alignment_score: int = Field(
        ..., ge=0, le=100, description="Aggregate alignment score"
    )
    scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-filter scores (0-100) keyed by filter name",
    )
    governance_questions: List[str] = Field(
        default_factory=list,
        description="Open governance questions if any",
    )
    execution_concerns: List[str] = Field(
        default_factory=list,
        description="Execution-related concerns if any",
    )
    recommendation: str = Field(
        default="",
        description="Doctrine-aligned recommendation or summary",
    )
    filter_details: Optional[Dict[str, FilterScore]] = Field(
        default=None,
        description="Optional per-filter reasoning and evidence",
    )

    def to_graph_state_dict(self) -> Dict[str, Any]:
        """Convert to dict suitable for GraphState.doctrine_evaluation."""
        return self.model_dump()


def empty_evaluation() -> DoctrineEvaluationResult:
    """Return a neutral evaluation when doctrine is skipped or fails."""
    return DoctrineEvaluationResult(
        institutional_alignment_score=0,
        scores={f: 0 for f in DOCTRINE_FILTERS},
        governance_questions=[],
        execution_concerns=[],
        recommendation="",
    )
