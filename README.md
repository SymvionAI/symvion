# Symvion AI Runtime

A multi-tenant AI orchestration framework powered by LangGraph.

## Core Capabilities
Symvion provides a clean, modular Python package that supports:
- Multi-tenant orchestration with isolated contexts
- Dynamic agent registration via APIs (stating payloads and expected responses schemas)
- Abstracted Tool/function calling 
- Observability hooks and state routing via LangGraph
- HITL / controlled GenUI pauses via `interrupt_before_tools` + `resume_stream` (package protocol; no HTTP layer)

## Installation

Create a project in one command (Python equivalent of `npx create-next-app`). uv and pipx download the CLI, scaffold the app, pin `symvion` in `requirements.txt`, create `.venv`, and install dependencies.

**uv**

```bash
uvx --from symvion create-symvion my-project
```

**pipx**

```bash
pipx run --spec symvion create-symvion my-project
```

Keep the CLI: `pipx install symvion`, then `create-symvion my-project`.

**pip** (Windows, macOS, Linux — no `PATH` setup required)

```bash
python -m pip install symvion
python -m symvion create my-project
```

**conda** — use `--no-install` so you do not nest a `.venv` inside the conda env:

```bash
conda create -n my-project python=3.12 -y
conda activate my-project
python -m pip install symvion
python -m symvion create my-project --no-install
cd my-project
python -m pip install -r requirements.txt
```

**Poetry**

```bash
python -m pip install symvion
python -m symvion create my-project --no-install
cd my-project
poetry init --no-interaction
poetry add symvion python-dotenv
```

`symvion init` is an alias for `create`. Pass `--no-install` to scaffold files only.

## Example Usage

```python
import asyncio
from symvion import Symvion, TenantConfig

async def main():
    config = TenantConfig(tenant_id="acme_corp")
    runtime = Symvion(config=config)
    
    runtime.register_agent({
        "name": "task_orchestrator",
        "description": "Orchestrates general tasks and queries",
        "system_prompt": "You are a helpful task orchestrator. Help the user with their request.",
        "input_schema": {
            "type": "object",
            "properties": {"task_description": {"type": "string"}}
        },
        "output_schema": {
            "type": "object",
            "properties": {"action_plan": {"type": "string"}, "is_complete": {"type": "boolean"}}
        },
        "tools": []
    })
    
    response = await runtime.chat(
        tenant="acme_corp",
        agent_name="task_orchestrator",
        payload_data={"task_description": "Organize my daily schedule and send notifications."}
    )
    print("Response:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## HITL / controlled GenUI (package protocol)

Symvion can pause before selected tools so a portal or widget can render a form, approve, edit args, reject, or supply a client-side result — then resume the same session.

**Allowlist only.** Pass `interrupt_before_tools=["render_signup_form"]` (or set `TenantConfig.interrupt_before_tools`). Empty list = off. Backend/MCP tools not on the list keep auto-running. Tools marked with `mark_client_tool(...)` always interrupt.

```python
from symvion.tools.hitl import mark_client_tool

async for event in runtime.chat_stream(
    tenant="acme_corp",
    message="Open the Signup form",
    session_id="thread-1",
    interrupt_before_tools=["render_signup_form"],
):
    if event["type"] == "tool_start":
        ...  # inspectors see this before interrupt
    if event["type"] == "interrupt":
        # event: tool, call_id, args, ui
        ...
    if event["type"] == "done" and event["status"] == "interrupted":
        break

async for event in runtime.resume_stream(
    tenant="acme_corp",
    session_id="thread-1",
    resume={"action": "provide_result", "result": {"symbol": "XYZ", "shares": 100}},
):
    print(event)
```

Resume actions: `approve` | `edit` (with `args`, client/UI tools only unless `trust_hitl_edit`) | `reject` | `provide_result` (with `result`).

**Security (0.4.10+):** `provide_result` and `edit` are only accepted for tools marked with `mark_client_tool(...)` unless you set `TenantConfig.trust_hitl_edit` (edit on server tools). `approve` uses the original interrupt args. Request metadata cannot override `interrupt_before_tools` / `user_role` / `agent_id` unless you set the matching `trust_client_*` flags after verifying the caller. `allowed_outbound_hosts` and `file_sandbox_dir` on `TenantConfig` are applied to claims and MCP clients. See [CHANGELOG.md](CHANGELOG.md).

Stream events: `token`, `agent_start`, `tool_start`, `interrupt`, `tool_end`, `metadata`, `done` (`success` | `interrupted`), `error`.

### Checkpointer note

Graphs compile with an injectable LangGraph checkpointer. The default is **process-local `MemorySaver`** — fine for package tests and single-process demos. Multi-replica or Coronation production should pass a durable checkpointer into `Symvion(config, checkpointer=...)`.
