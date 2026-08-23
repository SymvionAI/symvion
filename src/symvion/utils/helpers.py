from typing import List, Any, Dict, Optional
from symvion.core.context import TenantContext
from symvion.core.logger import logger

def ensure_messages(messages: List[Any]) -> List[Any]:
    """Ensures a list of messages is in a format compatible with LangChain models."""
    from langchain_core.messages import BaseMessage
    return [msg for msg in messages if isinstance(msg, BaseMessage)]

def filter_allowed_tools(context: TenantContext, tools: List[Any]) -> List[Any]:
    """
    Filters a list of tools based on the IAM policies provided in the TenantContext.
    Implementation of 'Active Governance'—preventing models from seeing tools they can't use.

    Empty ``iam_policies`` fails closed (no tools). Missing policy key also yields no tools
    unless the role is granted ``*``.
    """
    iam_policies = context.metadata.get("iam_policies")
    user_role = context.metadata.get("user_role", "user")

    # Fail closed: unset or empty policies mean no tools are exposed.
    if not iam_policies:
        logger.warning(
            "IAM_FAIL_CLOSED",
            context,
            role=user_role,
            reason="empty_or_missing_iam_policies",
        )
        return []

    allowed_tool_names = iam_policies.get(user_role, [])

    if "*" in allowed_tool_names:
        return tools

    filtered_tools = [
        t for t in tools if getattr(t, "name", None) in allowed_tool_names
    ]

    removed_count = len(tools) - len(filtered_tools)
    if removed_count > 0:
        logger.info("TOOLS_FILTERED_BY_IAM", context, role=user_role, removed=removed_count)

    return filtered_tools

def calculate_usage(usage: Dict[str, int]) -> int:
    """
    Calculates the total token usage.
    Returns: int (sum of input + output tokens)
    """
    in_tokens = usage.get("input_tokens", 0)
    out_tokens = usage.get("output_tokens", 0)
    return in_tokens + out_tokens

def merge_usage(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two usage dictionaries, summing numeric values.
    """
    result = {**dict1}
    for k, v in dict2.items():
        if k in result:
            if isinstance(v, dict) and isinstance(result[k], dict):
                result[k] = merge_usage(result[k], v)
            elif isinstance(v, (int, float)) and isinstance(result[k], (int, float)):
                result[k] += v
            else:
                # Type mismatch or non-numeric, overwrite with newer value
                result[k] = v
        else:
            result[k] = v
    return result
