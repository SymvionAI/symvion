"""
Doctrine evaluation engine: rule-based scoring against founder doctrine.
No LLM; fast path (<10ms) for low-complexity tasks.
"""

import logging
from typing import Dict, List, Any

from app.runtime.agents.legacy_agent.doctrine_models import (
    DoctrineEvaluationResult,
    DOCTRINE_FILTERS,
    FilterScore,
)
from app.runtime.agents.legacy_agent.doctrine_cache import get_principles

logger = logging.getLogger(__name__)

# Map doctrine filter names to heuristic keyword sets (for fast rule scoring)
_FILTER_KEYWORDS: Dict[str, List[str]] = {
    "institutional_strength": ["institution", "institutional", "structure", "structural", "governance", "policy", "process"],
    "governance_integrity": ["accountability", "audit", "traceable", "decision", "approval", "governance"],
    "execution_readiness": ["owner", "milestone", "deadline", "timeline", "deliverable", "action", "execute"],
    "strategic_leverage": ["strategy", "strategic", "long-term", "vision", "priority", "investment"],
    "talent_impact": ["talent", "people", "team", "capability", "skill", "training"],
    "capital_logic": ["capital", "investment", "return", "risk", "budget", "cost"],
    "ecosystem_relevance": ["ecosystem", "stakeholder", "market", "partner", "customer"],
    "vision_alignment": ["vision", "mission", "value", "align", "objective", "goal"],
}


def _normalize_document_view(context: Dict[str, Any], agent_response: str) -> str:
    """
    Build a single text view for doctrine evaluation from context and agent response.
    If context has document_extract (title, summary, sections, entities, key_points), use it.
    Otherwise use document_analysis and agent_response.
    """
    parts: List[str] = []
    extract = context.get("document_extract")
    if isinstance(extract, dict):
        if extract.get("title"):
            parts.append(str(extract["title"]))
        if extract.get("summary"):
            parts.append(str(extract["summary"]))
        for section in extract.get("sections") or []:
            if isinstance(section, dict) and section.get("content"):
                parts.append(str(section.get("content", "")))
            elif isinstance(section, str):
                parts.append(section)
        for entity in extract.get("entities") or []:
            if isinstance(entity, str):
                parts.append(entity)
            elif isinstance(entity, dict) and entity.get("name"):
                parts.append(str(entity.get("name", "")))
        for kp in extract.get("key_points") or []:
            if isinstance(kp, str):
                parts.append(kp)
        if parts:
            return "\n".join(parts)
    doc_analysis = context.get("document_analysis") or ""
    if doc_analysis:
        parts.append(doc_analysis)
    parts.append(agent_response or "")
    return "\n".join(p for p in parts if p)


def _score_filter(text: str, filter_name: str, principles: List[Dict[str, Any]]) -> FilterScore:
    """Heuristic score for one filter: keyword presence and rule mentions."""
    text_lower = (text or "").lower()
    keywords = _FILTER_KEYWORDS.get(filter_name, [])
    # Count keyword hits (each keyword at most once)
    hits = sum(1 for k in keywords if k in text_lower)
    max_hits = max(len(keywords), 1)
    raw = min(100, int((hits / max_hits) * 85) + 15)  # 15–100 band
    score = max(0, min(100, raw))
    evidence: List[str] = []
    for k in keywords:
        if k in text_lower:
            # Find a short snippet
            idx = text_lower.find(k)
            snippet = (text[idx : idx + 60] + "…") if idx >= 0 else k
            evidence.append(snippet.strip())
    reasoning = f"Fast rule check: {hits}/{len(keywords)} keyword signals."
    return FilterScore(score=score, reasoning=reasoning, evidence=evidence[:3])


def evaluate(
    agent_response: str,
    context: Dict[str, Any],
) -> DoctrineEvaluationResult:
    """
    Run doctrine evaluation using cached principles and heuristic scoring.
    No LLM; suitable for fast path (<10ms).

    Args:
        agent_response: The agent's response text.
        context: Graph context (document_analysis, optional document_extract).

    Returns:
        DoctrineEvaluationResult with institutional_alignment_score, scores, etc.
    """
    principles = get_principles()
    text = _normalize_document_view(context, agent_response or "")

    filter_details: Dict[str, FilterScore] = {}
    scores: Dict[str, int] = {}

    for f in DOCTRINE_FILTERS:
        detail = _score_filter(text, f, principles)
        filter_details[f] = detail
        scores[f] = detail.score

    if scores:
        institutional_alignment_score = int(sum(scores.values()) / len(scores))
    else:
        institutional_alignment_score = 0

    governance_questions: List[str] = []
    execution_concerns: List[str] = []
    if scores.get("governance_integrity", 100) < 50:
        governance_questions.append("Ensure governance and accountability are explicit.")
    if scores.get("execution_readiness", 100) < 50:
        execution_concerns.append("Consider adding owners and milestones.")

    recommendation = ""
    if institutional_alignment_score >= 70:
        recommendation = "Output aligns with doctrine principles."
    elif institutional_alignment_score >= 40:
        recommendation = "Consider strengthening alignment with institutional and execution discipline."
    else:
        recommendation = "Recommend review against founder doctrine before finalizing."

    return DoctrineEvaluationResult(
        institutional_alignment_score=institutional_alignment_score,
        scores=scores,
        governance_questions=governance_questions,
        execution_concerns=execution_concerns,
        recommendation=recommendation,
        filter_details=filter_details,
    )
