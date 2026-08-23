# Changelog

All notable changes to Symvion are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.10] — 2026-08-15

### Security

Package-owned follow-up: safer defaults with **no new required host AuthN**. Hosts still authenticate at the gateway. Existing `chat()` / `resume_stream()` signatures are unchanged.

- **HITL `edit`** — Client-supplied args on `edit` are accepted only for `mark_client_tool(...)` tools. Backend tools on the interrupt allowlist stay `approve` / `reject`. Hosts that need edit on server tools may set `TenantConfig.trust_hitl_edit`.
- **Interrupt allowlist** — Request metadata can no longer override `interrupt_before_tools`, `tenant_id`, or `trust_hitl_edit`. Hosts still pass the allowlist as the `chat()` / `chat_stream()` kwarg or via `TenantConfig`.
- **Config wiring** — `allowed_outbound_hosts` and `file_sandbox_dir` are passed into claims tools and MCP clients (empty allowlist keeps current behavior).
- **SSRF** — Block IPv4-mapped IPv6, decimal IPv4, and known metadata hostnames.
- **Claims / quote tools** — Execute through `ToolSafetyWrapper`; HITL `GraphInterrupt` is no longer swallowed as a tool error.
- **Object tools** — `ToolRegistry.execute()` applies IAM when a `TenantContext` is provided.
- **Claim document URLs** — Validated with the same outbound URL helper before POST.

### Changed
- New `TenantConfig.trust_hitl_edit` (default `False`).
- `pyproject.toml` floors: cryptography, urllib3, idna, PyJWT. LangChain major stays `>=0.3.0` so existing 0.3 hosts are not forced onto 1.x.
- `requirements.txt` adds `langgraph-checkpoint>=4.1.1` for the host runtime.

### Migration notes (hosts)
1. GenUI forms that use `edit` should already call `mark_client_tool(tool)` — no change.
2. If you interrupt **server** tools and resume with `{"action": "edit", ...}`, set `trust_hitl_edit=True` after you authenticate resume.
3. Optional: set `allowed_outbound_hosts` / `file_sandbox_dir` on `TenantConfig`; they now take effect.

### Tests
- HITL `edit` denied for server tools; allowed with `trust_hitl_edit` or client tools.
- Metadata cannot override the interrupt allowlist.
- Mapped-IPv6 / decimal-IPv4 / metadata host URL rejection.

## [0.4.9] — 2026-08-15

### Security

Package-owned hardening from a VAPT / static review. Host applications remain responsible for AuthN/AuthZ at the gateway; these changes make insecure defaults harder to misuse.

#### Critical / High
- **Secret logging** — Claims tools no longer log auth tokens in plaintext; cache keys use a SHA-256 fingerprint instead of the raw token.
- **HITL resume spoofing** — `provide_result` is allowed only for tools marked with `mark_client_tool(...)`. `approve` always executes with the original interrupt args (client-supplied args on approve are ignored).
- **Client metadata trust** — `user_role` and `agent_id` from request metadata are ignored by default. Hosts that map verified JWT/gateway claims may opt in with:
  - `TenantConfig.trust_client_role`
  - `TenantConfig.trust_client_agent_id`
- **Session isolation** — LangGraph `thread_id` is now `{tenant_id}:{session_id}` to reduce cross-tenant checkpoint collisions on shared savers.
- **SSRF / redirect abuse** — HTTP clients use `follow_redirects=False`. Outbound URLs are validated (scheme checks; private/link-local/metadata hosts blocked). Optional `allowed_outbound_hosts`.
- **MCP local file read** — `file_path` reads are sandboxed under `TenantConfig.file_sandbox_dir` or `SYMVION_FILE_SANDBOX` (temp dir fallback). Path traversal outside the sandbox is rejected.
- **Process env bleed** — Arbitrary `TenantConfig.env_vars` are no longer written into process-global `os.environ`. Only known provider credential keys are applied, and only when unset.
- **IAM fail-closed** — Empty / missing IAM policies no longer expose all tools. Dynamic agent tools are filtered and executed through `ToolSafetyWrapper`.

#### Medium / defense in depth
- Client-facing stream/request errors are generic; details stay in server logs.
- Hallucination middleware fails closed (unverified → score `0.0`) instead of defaulting to “safe”.
- Tool safety wrapper applies timeouts to sync tools and avoids logging raw tool argument values.
- Warning when HITL allowlists are configured without an injectable durable checkpointer.

### Changed
- New helpers in `symvion.utils.security`.
- New `TenantConfig` fields: `trust_client_role`, `trust_client_agent_id`, `default_user_role`, `allowed_outbound_hosts`, `file_sandbox_dir`.
- Dependency floors raised in `pyproject.toml` and `requirements.txt` (notably cryptography, Pillow, PyJWT, urllib3, langsmith, langchain).

### Migration notes (hosts)
1. Reinstall / refresh the lockfile from updated dependency floors.
2. For GenUI tools that use `resume={"action": "provide_result", ...}`, call `mark_client_tool(tool)`.
3. After authenticating callers, set `trust_client_role` / `trust_client_agent_id` only when metadata is server-derived.
4. Pass a durable LangGraph checkpointer in multi-replica HITL deployments: `Symvion(config, checkpointer=...)`.
5. If you relied on dumping non-provider secrets via `env_vars`, inject those outside Symvion or use provider constructor config.

### Tests
- Added `tests/test_security_helpers.py`.
- Updated `tests/test_hitl_resume.py` for client-tool `provide_result` rules and trust flags.

## [0.4.8] — prior

Previous packaged release baseline before this security hardening pass.
