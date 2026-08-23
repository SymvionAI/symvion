"""
DoctrineRouter: decides whether to use fast doctrine rules or LegacyAgent.
Rule-based classification; no LLM.
"""

import logging
from typing import Dict, Any, Literal

logger = logging.getLogger(__name__)

RouteDecision = Literal["FAST_RULES", "LEGACY_REASONING"]

# Task types that are always low complexity -> fast rules
_LOW_COMPLEXITY_AGENT_TYPES = frozenset({
    "complaint",
    "billing",
    "registration",
    "default",
    "default_agent",
    "insurance",
    "quote_generator",
})

# Keywords in context or response that suggest high-complexity strategic content
_HIGH_COMPLEXITY_KEYWORDS = frozenset({
    "strategy", "strategic", "board", "board paper", "memo", "acquisition",
    "acquire", "merger", "transformation", "organizational change",
    "governance change", "governance reform", "restructure", "restructuring",
    "board meeting", "executive committee", "capex", "strategic initiative",
})

# Explicit context flag for high-complexity (e.g. set by upstream)
_CONTEXT_FLAG_HIGH = "doctrine_requires_legacy"
_CONTEXT_FLAG_DOCUMENT_CLASS = "document_class"


def route(context: Dict[str, Any], task_type: str) -> RouteDecision:
    """
    Decide whether to use fast doctrine rules or LegacyAgent (strategic reasoning).

    LOW complexity -> FAST_RULES: customer support, product descriptions,
    simple internal communications.

    HIGH complexity -> LEGACY_REASONING: strategy memos, board papers,
    acquisitions, organizational transformation, governance changes.

    Args:
        context: Graph context (may contain document_analysis, document_class, etc.).
        task_type: Agent type or task identifier (e.g. from state.agent_type).

    Returns:
        "FAST_RULES" or "LEGACY_REASONING".
    """
    task_type_normalized = (task_type or "").strip().lower()

    # Explicit override from context
    if context.get(_CONTEXT_FLAG_HIGH) is True:
        logger.info("[DOCTRINE_ROUTER] context flag doctrine_requires_legacy=True -> LEGACY_REASONING")
        return "LEGACY_REASONING"

    doc_class = context.get(_CONTEXT_FLAG_DOCUMENT_CLASS)
    if isinstance(doc_class, str) and doc_class.lower() in (
        "strategy", "board", "governance", "acquisition", "transformation"
    ):
        logger.info("[DOCTRINE_ROUTER] document_class=%s -> LEGACY_REASONING", doc_class)
        return "LEGACY_REASONING"

    # Scan text for high-complexity keywords (before agent-type check so strategic
    # user requests like "strategy memo for the board" trigger LEGACY_REASONING
    # even when handled by default_agent)
    text_parts = [
        str(context.get("document_analysis", "")),
        str(context.get("agent_response", "")),
        str(context.get("user_message", "")),
    ]
    text = " ".join(text_parts).lower()
    for kw in _HIGH_COMPLEXITY_KEYWORDS:
        if kw in text:
            logger.info("[DOCTRINE_ROUTER] keyword '%s' in context -> LEGACY_REASONING", kw)
            return "LEGACY_REASONING"

    # Agent type is low complexity -> fast path
    if task_type_normalized in _LOW_COMPLEXITY_AGENT_TYPES:
        logger.info("[DOCTRINE_ROUTER] task_type=%s -> FAST_RULES", task_type_normalized)
        return "FAST_RULES"

    # claims and document_intelligence: default to fast path unless context suggests otherwise
    if task_type_normalized in ("claims", "document_intelligence"):
        logger.info("[DOCTRINE_ROUTER] task_type=%s (no strategic signals) -> FAST_RULES", task_type_normalized)
        return "FAST_RULES"

    # Unknown or future agent types: default to fast rules to avoid unnecessary LLM
    logger.info("[DOCTRINE_ROUTER] task_type=%s default -> FAST_RULES", task_type_normalized)
    return "FAST_RULES"
