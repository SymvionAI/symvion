from types import SimpleNamespace

from symvion.tools.registry import ToolRegistry
from symvion.tools.resolution import load_tools_by_name, normalize_tool_calls


def test_normalize_recovers_invalid_json_args_and_keeps_parse_errors():
    response = SimpleNamespace(
        tool_calls=[{"name": "submit_claim", "args": {"policy_number": "1"}, "id": "call_ok"}],
        invalid_tool_calls=[
            {"name": "save_quote", "args": '{"customer_email": "a@b.c"}', "id": "call_recovered", "error": None},
            {"name": "broken", "args": "{", "id": "call_bad", "error": "bad json"},
        ],
    )

    calls = normalize_tool_calls(response)

    assert calls[0]["name"] == "submit_claim"
    assert calls[0]["error"] is None
    assert calls[1]["name"] == "save_quote"
    assert calls[1]["args"] == {"customer_email": "a@b.c"}
    assert calls[1]["error"] is None
    assert calls[2]["name"] == "broken"
    assert calls[2]["error"] == "bad json"


def test_registry_lookup_binds_callable_and_langchain_tool():
    registry = ToolRegistry()

    def ping(value: str) -> str:
        """Return the value."""
        return value

    registry.register("ping", ping)
    bound = registry.lookup_bindable("ping")
    assert bound is not None
    assert bound.name == "ping"

    class _Marker:
        name = "mcp_proxy"

    marker = _Marker()
    registry.register_langchain_tool("remote_lookup", marker)
    assert registry.lookup_bindable("remote_lookup") is marker

    loaded = load_tools_by_name(["ping", "remote_lookup", "missing"], registry)
    assert [tool.name for tool in loaded] == ["ping", "mcp_proxy"]
    assert loaded[1] is marker
