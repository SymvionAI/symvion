"""Tests for HITL allowlist gate and interrupt/resume streaming protocol."""

import asyncio
from typing import Any, Dict, List, Optional, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from symvion.tools.hitl import (
    HitlDecisionError,
    apply_hitl_decision,
    is_client_tool,
    mark_client_tool,
    run_tool_with_hitl,
    should_interrupt_tool,
)


def test_allowlist_empty_does_not_interrupt():
    assert should_interrupt_tool("backend_mcp", []) is False
    assert should_interrupt_tool("backend_mcp", None) is False


def test_allowlist_only_named_tools():
    assert should_interrupt_tool("render_ipo_form", ["render_ipo_form"]) is True
    assert should_interrupt_tool("submit_claim", ["render_ipo_form"]) is False


def test_client_tool_marker_always_interrupts():
    class T:
        name = "ui_form"

    tool = mark_client_tool(T())
    assert is_client_tool(tool)
    assert should_interrupt_tool("ui_form", [], tool=tool) is True


@pytest.mark.asyncio
async def test_apply_hitl_decisions():
    async def execute(**kwargs):
        return {"ran": kwargs}

    approved = await apply_hitl_decision({"action": "approve"}, {"a": 1}, execute)
    assert approved == {"ran": {"a": 1}}

    # approve must ignore client-supplied args
    approved_ignore = await apply_hitl_decision(
        {"action": "approve", "args": {"a": 999}}, {"a": 1}, execute
    )
    assert approved_ignore == {"ran": {"a": 1}}

    edited = await apply_hitl_decision(
        {"action": "edit", "args": {"a": 2}},
        {"a": 1},
        execute,
        client_tool=True,
    )
    assert edited == {"ran": {"a": 2}}

    with pytest.raises(HitlDecisionError):
        await apply_hitl_decision(
            {"action": "edit", "args": {"a": 2}}, {"a": 1}, execute, client_tool=False
        )

    server_edit = await apply_hitl_decision(
        {"action": "edit", "args": {"a": 3}},
        {"a": 1},
        execute,
        client_tool=False,
        allow_server_edit=True,
    )
    assert server_edit == {"ran": {"a": 3}}

    rejected = await apply_hitl_decision({"action": "reject"}, {"a": 1}, execute)
    assert "rejected" in str(rejected).lower() or rejected == "User rejected tool execution"

    with pytest.raises(HitlDecisionError):
        await apply_hitl_decision(
            {"action": "provide_result", "result": {"form": "ok"}},
            {"a": 1},
            execute,
            client_tool=False,
        )

    provided = await apply_hitl_decision(
        {"action": "provide_result", "result": {"form": "ok"}},
        {"a": 1},
        execute,
        client_tool=True,
    )
    assert provided == {"form": "ok"}


class _MiniState(TypedDict):
    value: str
    tool_result: Optional[Any]


@pytest.mark.asyncio
async def test_langgraph_interrupt_resume_with_allowlist():
    """Minimal graph: HITL tool pauses; backend tool would not (unit path via helper)."""
    executed: List[str] = []

    class _ClientForm:
        name = "render_ipo_form"

    client_form = mark_client_tool(_ClientForm())

    async def node(state: _MiniState, config: RunnableConfig) -> _MiniState:
        async def _backend(**kwargs):
            executed.append("backend")
            return "backend_ok"

        async def _genui(**kwargs):
            executed.append("genui")
            return "genui_ok"

        # Backend not on allowlist — runs immediately
        backend = await run_tool_with_hitl(
            tool_name="backend_mcp",
            tool_args={},
            call_id="c1",
            execute=_backend,
            config=config,
        )
        # Client/UI tool — interrupts; provide_result allowed
        genui = await run_tool_with_hitl(
            tool_name="render_ipo_form",
            tool_args={"fields": ["symbol"]},
            call_id="c2",
            execute=_genui,
            config=config,
            tool=client_form,
        )
        return {**state, "value": f"{backend}|{genui}", "tool_result": genui}

    workflow = StateGraph(_MiniState)
    workflow.add_node("work", node)
    workflow.set_entry_point("work")
    workflow.add_edge("work", END)
    graph = workflow.compile(checkpointer=MemorySaver())

    config = {
        "configurable": {"thread_id": "hitl-test-1"},
        "metadata": {"interrupt_before_tools": ["render_ipo_form"]},
    }

    await graph.ainvoke({"value": "", "tool_result": None}, config=config)
    snap = graph.get_state(config)
    interrupts = list(getattr(snap, "interrupts", ()) or ())
    if not interrupts:
        for task in getattr(snap, "tasks", ()) or ():
            interrupts.extend(list(getattr(task, "interrupts", ()) or ()))
    assert interrupts, "expected HITL interrupt on render_ipo_form"
    assert executed == ["backend"]  # genui not executed yet

    payload = getattr(interrupts[0], "value", interrupts[0])
    assert payload["tool"] == "render_ipo_form"
    assert payload.get("client_tool") is True

    result = await graph.ainvoke(
        Command(resume={"action": "provide_result", "result": {"symbol": "XYZ"}}),
        config=config,
    )
    assert result["tool_result"] == {"symbol": "XYZ"}
    assert "backend_ok" in result["value"]


@pytest.mark.asyncio
async def test_symvion_stream_interrupt_event_order(monkeypatch):
    """Symvion chat_stream emits tool_start before interrupt when allowlisted."""
    import os
    from symvion import Symvion, TenantConfig
    from symvion.tools.hitl import mark_client_tool

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")

    class _FormTool:
        name = "render_ipo_form"

    form_tool = mark_client_tool(_FormTool())

    class FormAgent:
        name = "form"
        description = "form"

        async def process(self, input_data, config=None):
            result = await run_tool_with_hitl(
                tool_name="render_ipo_form",
                tool_args={"title": "IPO"},
                call_id="form-1",
                execute=lambda **kw: {"ok": True},
                config=config,
                tool=form_tool,
            )
            return {"agent_response": f"done:{result}", "token_usage": {}}

    config = TenantConfig(
        tenant_id="hitl_tenant",
        enabled_agents=[],
        interrupt_before_tools=[],
        logging={"enabled": False},
        router_type="keyword",
        trust_client_agent_id=True,
    )
    runtime = Symvion(config=config, checkpointer=MemorySaver())
    runtime.register_agent(FormAgent())

    events = []
    async for evt in runtime.chat_stream(
        tenant="hitl_tenant",
        message="show form",
        session_id="s-hitl-stream",
        metadata={"agent_id": "form"},
        interrupt_before_tools=["render_ipo_form"],
    ):
        events.append(evt)

    types = [e["type"] for e in events]
    assert "interrupt" in types, f"events={events}"
    assert "done" in types
    assert events[-1]["status"] == "interrupted"
    # tool_start should appear before interrupt
    assert types.index("tool_start") < types.index("interrupt")

    resume_events = []
    async for evt in runtime.resume_stream(
        tenant="hitl_tenant",
        session_id="s-hitl-stream",
        resume={"action": "provide_result", "result": {"filled": True}},
    ):
        resume_events.append(evt)

    assert any(
        e.get("type") == "done" and e.get("status") == "success" for e in resume_events
    ), f"resume_events={resume_events}"
