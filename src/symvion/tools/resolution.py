"""Resolve agent tool names and normalize model tool calls."""

from __future__ import annotations

import importlib
import json
import pkgutil
from typing import Any, Dict, List, Optional


def _field(tool_call: Any, key: str) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def _parse_args(raw_args: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    Return every model tool call the agent should act on.

    Parsed calls come from ``tool_calls``. Calls that only landed on
    ``invalid_tool_calls`` are recovered when their arguments are a JSON
    object; otherwise they carry an error so the model can retry.
    """
    normalized: List[Dict[str, Any]] = []
    for index, tool_call in enumerate(getattr(response, "tool_calls", None) or []):
        name = _field(tool_call, "name")
        args = _parse_args(_field(tool_call, "args")) or {}
        normalized.append(
            {
                "name": name,
                "args": args,
                "id": _field(tool_call, "id") or f"call_{name}_{index}",
                "error": None,
            }
        )

    offset = len(normalized)
    for index, tool_call in enumerate(getattr(response, "invalid_tool_calls", None) or []):
        name = _field(tool_call, "name")
        parsed = _parse_args(_field(tool_call, "args"))
        error = _field(tool_call, "error")
        recoverable = bool(name) and parsed is not None
        normalized.append(
            {
                "name": name,
                "args": parsed or {},
                "id": _field(tool_call, "id") or f"call_{name or 'invalid'}_{offset + index}",
                "error": None if recoverable else (error or "Tool call could not be parsed"),
            }
        )
    return normalized


def load_tools_by_name(names: List[Any], registry: Any = None) -> List[Any]:
    """
    Resolve tool names to LangChain tools.

    A local ``tools`` package is checked first. Names that are not there are
    taken from the runtime tool registry, including LangChain tools registered
    with ``register_langchain_tool``.
    """
    wanted: List[str] = []
    seen = set()
    objects: List[Any] = []
    for item in names or []:
        if isinstance(item, str):
            if item not in seen:
                seen.add(item)
                wanted.append(item)
        elif getattr(item, "name", None) and item.name not in seen:
            seen.add(item.name)
            objects.append(item)

    found: Dict[str, Any] = {}
    if wanted:
        try:
            import tools as tools_pkg
        except ImportError:
            tools_pkg = None
        if tools_pkg is not None and hasattr(tools_pkg, "__path__"):
            for _, modname, _ in pkgutil.iter_modules(tools_pkg.__path__):
                try:
                    module = importlib.import_module(f"tools.{modname}")
                except Exception:
                    continue
                for name in wanted:
                    if name in found or not hasattr(module, name):
                        continue
                    candidate = getattr(module, name)
                    if getattr(candidate, "name", None):
                        found[name] = candidate

    if registry is not None and hasattr(registry, "lookup_bindable"):
        for name in wanted:
            if name in found:
                continue
            candidate = registry.lookup_bindable(name)
            if candidate is not None:
                found[name] = candidate

    have = {getattr(item, "name", None) for item in objects}
    return objects + [found[name] for name in wanted if name in found and name not in have]
