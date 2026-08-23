"""
Helper utilities for common operations.
Provides reusable functions for the AI runtime.
"""
from typing import Dict, Any
from datetime import datetime


def generate_conversation_id(tenant_id: str) -> str:
    """
    Generate a unique conversation ID.

    Args:
        tenant_id: Unique tenant identifier

    Returns:
        Unique conversation identifier
    """
    timestamp = datetime.utcnow().isoformat()
    return f"{tenant_id}-{timestamp}"


def sanitize_message(message: str) -> str:
    """
    Sanitize user message input.

    Args:
        message: Raw message input

    Returns:
        Sanitized message
    """
    # Basic sanitization - can be extended
    return message.strip()


def format_response(
    tenant_id: str, conversation_id: str, content: str, metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Format a standardized response.

    Args:
        tenant_id: Unique tenant identifier
        conversation_id: Unique conversation identifier
        content: Response content
        metadata: Additional response metadata

    Returns:
        Formatted response dictionary
    """
    return {
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "content": content,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat(),
    }
