"""
Logging utilities for LangSmith integration.
Handles observability and tracing for AI operations.
"""
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def log_to_langsmith(
    tenant_id: str,
    conversation_id: str,
    agent_type: str,
    tool_calls: Optional[list] = None,
    escalation_flag: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Log operation to LangSmith with tenant and conversation context.

    Args:
        tenant_id: Unique tenant identifier
        conversation_id: Unique conversation identifier
        agent_type: Type of agent that processed the request
        tool_calls: List of tool calls made during processing
        escalation_flag: Whether the request was escalated to human agent
        metadata: Additional metadata for logging
    """
    log_data = {
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "agent_type": agent_type,
        "tool_calls": tool_calls or [],
        "escalation_flag": escalation_flag,
        "metadata": metadata or {},
    }
    # LangSmith logging logic will be implemented here
    logger.info(f"LangSmith log: {log_data}")


def emit_doctrine_event(
    event_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a doctrine-layer telemetry event for observability.
    Events: legacy_agent.invoked, doctrine_router.route, doctrine_alignment.scored.
    Metadata should include at least: document_id (if present), alignment_score,
    decision_type (FAST_RULES | LEGACY_REASONING), latency_ms.
    """
    payload = {"event": event_name, "metadata": metadata or {}}
    logger.info(f"Doctrine event: {payload}")
