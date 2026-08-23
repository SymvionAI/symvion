"""
HITL gate for controlled GenUI tool pauses.

Only tools on the ``interrupt_before_tools`` allowlist (or marked as client/UI
tools) pause via LangGraph ``interrupt()``. Backend/MCP tools not on the list
continue to auto-execute.

``provide_result`` is restricted to client/UI-marked tools so a resume payload
cannot spoof outcomes for server-side tools that only paused for approval.
``edit`` is likewise restricted unless the host sets ``trust_hitl_edit``.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

CLIENT_TOOL_TAG = "client_tool"
SYMVION_TOOL_START = "symvion_tool_start"
SYMVION_TOOL_END = "symvion_tool_end"

ExecuteFn = Callable[..., Union[Any, Awaitable[Any]]]


class HitlDecisionError(ValueError):
    """Raised when a resume payload is invalid or unauthorized for the tool."""


def mark_client_tool(tool: Any) -> Any:
    """Mark a LangChain tool (or similar) as a client/UI tool — always interruptible."""
    try:
        setattr(tool, "client_tool", True)
    except Exception:
        pass
    tags = list(getattr(tool, "tags", None) or [])
    if CLIENT_TOOL_TAG not in tags:
        tags.append(CLIENT_TOOL_TAG)
        try:
            tool.tags = tags
        except Exception:
            pass
    meta = dict(getattr(tool, "metadata", None) or {})
    meta["client_tool"] = True
    try:
        tool.metadata = meta
    except Exception:
        pass
    return tool


def is_client_tool(tool: Any = None) -> bool:
    """Return True if the tool is marked for client/UI (GenUI) execution."""
    if tool is None:
        return False
    if getattr(tool, "client_tool", False):
        return True
    tags = getattr(tool, "tags", None) or []
    if CLIENT_TOOL_TAG in tags or "client" in tags or "client_ui" in tags:
        return True
    meta = getattr(tool, "metadata", None) or {}
    if meta.get("client_tool") or meta.get("client_ui"):
        return True
    return False


def should_interrupt_tool(
    tool_name: str,
    interrupt_before_tools: Optional[List[str]],
    tool: Any = None,
) -> bool:
    """True only for allowlisted names or client/UI-marked tools."""
    if is_client_tool(tool):
        return True
    if not interrupt_before_tools:
        return False
    return tool_name in interrupt_before_tools


def get_interrupt_allowlist(config: Optional[RunnableConfig]) -> List[str]:
    if not config:
        return []
    meta = (config.get("metadata") if isinstance(config, dict) else None) or {}
    allow = meta.get("interrupt_before_tools")
    if allow is None:
        return []
    if isinstance(allow, str):
        return [allow]
    return list(allow)


async def _emit_custom(name: str, data: Dict[str, Any], config: Optional[RunnableConfig]) -> None:
    if not config:
        return
    try:
        from langchain_core.callbacks.manager import adispatch_custom_event

        await adispatch_custom_event(name, data, config=config)
    except Exception:
        # Streaming inspectors may not have a callback manager; gate still works.
        pass


async def _do_execute(execute: ExecuteFn, tool_args: Dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(execute):
        try:
            return await execute(**tool_args)
        except TypeError:
            return await execute(tool_args)
    result = execute(**tool_args) if tool_args is not None else execute()
    if inspect.isawaitable(result):
        return await result
    return result


def reraise_if_hitl_pause(exc: BaseException) -> None:
    """Re-raise LangGraph interrupt exceptions so HITL can surface to the host."""
    try:
        from langgraph.errors import GraphInterrupt
    except ImportError:  # pragma: no cover
        GraphInterrupt = ()  # type: ignore
    if GraphInterrupt and isinstance(exc, GraphInterrupt):
        raise exc
    if type(exc).__name__ in ("GraphInterrupt", "GraphBubbleUp"):
        raise exc


async def apply_hitl_decision(
    decision: Any,
    original_args: Dict[str, Any],
    execute: ExecuteFn,
    *,
    client_tool: bool = False,
    allow_server_edit: bool = False,
) -> Any:
    """
    Apply a resume payload from the client.

    - ``approve`` runs with original (server) args only.
    - ``edit`` runs with client-supplied args only for client/UI tools, or when
      the host sets ``TenantConfig.trust_hitl_edit``.
    - ``reject`` skips execution.
    - ``provide_result`` is allowed only for client/UI-marked tools.
    """
    if decision is None:
        return await _do_execute(execute, original_args)
    if not isinstance(decision, dict):
        raise HitlDecisionError("HITL resume payload must be a dict")

    action = decision.get("action", "approve")
    if action == "reject":
        return decision.get("result", "User rejected tool execution")
    if action == "provide_result":
        if not client_tool:
            raise HitlDecisionError(
                "provide_result is only allowed for client/UI-marked tools"
            )
        return decision.get("result")
    if action == "edit":
        if not client_tool and not allow_server_edit:
            raise HitlDecisionError(
                "edit is only allowed for client/UI-marked tools"
            )
        return await _do_execute(execute, decision.get("args", original_args) or {})
    if action == "approve":
        # Never accept client-supplied args on approve — use original interrupt args.
        return await _do_execute(execute, original_args)
    raise HitlDecisionError(f"Unsupported HITL action: {action}")


async def run_tool_with_hitl(
    *,
    tool_name: str,
    tool_args: Dict[str, Any],
    call_id: str,
    execute: ExecuteFn,
    config: Optional[RunnableConfig] = None,
    tool: Any = None,
    ui: Any = None,
) -> Any:
    """
    Emit ``tool_start``, optionally ``interrupt()`` for GenUI tools, then execute.

    Resume decisions: approve | edit | reject | provide_result (client tools only).
    """
    allowlist = get_interrupt_allowlist(config)
    args = dict(tool_args or {})
    client = is_client_tool(tool)
    meta = (config.get("metadata") if isinstance(config, dict) else None) or {}
    allow_server_edit = bool(meta.get("trust_hitl_edit"))

    await _emit_custom(
        SYMVION_TOOL_START,
        {"tool": tool_name, "inputs": args, "call_id": call_id},
        config,
    )

    error: Optional[str] = None
    try:
        if should_interrupt_tool(tool_name, allowlist, tool):
            decision = interrupt(
                {
                    "reason": "tool_approval",
                    "tool": tool_name,
                    "call_id": call_id,
                    "args": args,
                    "ui": ui,
                    "client_tool": client,
                }
            )
            result = await apply_hitl_decision(
                decision,
                args,
                execute,
                client_tool=client,
                allow_server_edit=allow_server_edit,
            )
        else:
            result = await _do_execute(execute, args)
    except Exception as exc:
        # Do not treat LangGraph HITL pauses as tool failures.
        reraise_if_hitl_pause(exc)
        error = str(exc)
        await _emit_custom(
            SYMVION_TOOL_END,
            {"tool": tool_name, "call_id": call_id, "output": None, "error": error},
            config,
        )
        raise

    await _emit_custom(
        SYMVION_TOOL_END,
        {"tool": tool_name, "call_id": call_id, "output": result, "error": None},
        config,
    )
    return result
