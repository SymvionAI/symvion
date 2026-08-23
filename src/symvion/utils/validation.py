"""
Validation utilities for tenant and tool validation.
Ensures tenant isolation and tool allowlist enforcement.
"""
from typing import List, Dict, Any


def validate_tenant_access(tenant_id: str, allowed_tenants: List[str]) -> bool:
    """
    Validate that a tenant has access to the system.

    Args:
        tenant_id: Unique tenant identifier
        allowed_tenants: List of allowed tenant identifiers

    Returns:
        True if tenant is allowed, False otherwise
    """
    return tenant_id in allowed_tenants


def validate_tool_access(tool_name: str, allowed_tools: List[str]) -> bool:
    """
    Validate that a tool is allowed for the current context.

    Args:
        tool_name: Name of the tool to validate
        allowed_tools: List of allowed tool identifiers

    Returns:
        True if tool is allowed, False otherwise
    """
    return tool_name in allowed_tools


def validate_agent_access(agent_type: str, allowed_agents: List[str]) -> bool:
    """
    Validate that an agent type is allowed for the current context.

    Args:
        agent_type: Type of agent to validate
        allowed_agents: List of allowed agent types

    Returns:
        True if agent is allowed, False otherwise
    """
    return agent_type in allowed_agents
