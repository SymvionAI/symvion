"""
Kafka topic definitions and naming conventions.
Centralizes topic names for consistency.
"""

# Topic naming pattern: {tenant_id}.conversation.request
# Example: tenant-123.conversation.request
# Backend sends to: {tenant_id}.conversation.request
# AI Runtime responds to: conversation.ai.response (general topic)


def get_request_topic(tenant_id: str) -> str:
    """
    Get the request topic name for a tenant.

    Args:
        tenant_id: Unique tenant identifier

    Returns:
        Kafka topic name for tenant requests
    """
    return f"{tenant_id}.conversation.request"


def get_response_topic() -> str:
    """
    Get the response topic name for AI responses.
    This is a general topic that all tenants use.

    Returns:
        Kafka topic name for AI responses
    """
    return "conversation.ai.response"


def get_escalation_topic(tenant_id: str) -> str:
    """
    Get the escalation topic name for a tenant.

    Args:
        tenant_id: Unique tenant identifier

    Returns:
        Kafka topic name for human agent escalations
    """
    return f"{tenant_id}.escalation.request"
