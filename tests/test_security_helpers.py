"""Unit tests for package security helpers."""

import os
from pathlib import Path

import pytest

from symvion.utils.security import (
    apply_provider_env_vars,
    hash_secret,
    redact_secret,
    resolve_sandboxed_path,
    sanitize_request_metadata,
    validate_outbound_url,
)


def test_hash_and_redact_secret():
    assert hash_secret(None) == "none"
    assert hash_secret("tok") != "tok"
    assert "present:" in redact_secret("secret-token")
    assert redact_secret(None) == "absent"


def test_sanitize_metadata_strips_untrusted_fields():
    meta = sanitize_request_metadata(
        {"user_role": "tenant_admin", "agent_id": "claims", "foo": 1},
        trust_client_role=False,
        trust_client_agent_id=False,
    )
    assert meta["user_role"] == "user"
    assert "agent_id" not in meta
    assert meta["foo"] == 1

    trusted = sanitize_request_metadata(
        {"user_role": "tenant_admin", "agent_id": "claims"},
        trust_client_role=True,
        trust_client_agent_id=True,
    )
    assert trusted["user_role"] == "tenant_admin"
    assert trusted["agent_id"] == "claims"

    stripped = sanitize_request_metadata(
        {
            "interrupt_before_tools": ["submit_claim"],
            "trust_hitl_edit": True,
            "iam_policies": {"user": ["*"]},
            "tenant_id": "other",
            "foo": 1,
        }
    )
    assert "interrupt_before_tools" not in stripped
    assert "trust_hitl_edit" not in stripped
    assert "iam_policies" not in stripped
    assert "tenant_id" not in stripped
    assert stripped["foo"] == 1


def test_validate_outbound_url_blocks_private():
    assert validate_outbound_url("https://example.com/api")
    with pytest.raises(ValueError):
        validate_outbound_url("https://169.254.169.254/latest/meta-data")
    with pytest.raises(ValueError):
        validate_outbound_url("ftp://example.com/x")
    # localhost allowed for local backends
    assert validate_outbound_url("http://localhost:3000", allow_localhost=True)
    with pytest.raises(ValueError):
        validate_outbound_url("https://[::ffff:169.254.169.254]/latest/meta-data")
    with pytest.raises(ValueError):
        validate_outbound_url("http://2130706433/")
    with pytest.raises(ValueError):
        validate_outbound_url("http://metadata.google.internal/")


def test_resolve_sandboxed_path(tmp_path: Path):
    safe = tmp_path / "doc.txt"
    safe.write_text("ok", encoding="utf-8")
    resolved = resolve_sandboxed_path(str(safe.name), sandbox_dir=str(tmp_path))
    assert resolved == safe.resolve()

    with pytest.raises(ValueError):
        resolve_sandboxed_path("../etc/passwd", sandbox_dir=str(tmp_path))


def test_apply_provider_env_vars_only_allowlisted(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVIL_KEY", raising=False)
    applied = apply_provider_env_vars(
        {"OPENAI_API_KEY": "sk-test", "EVIL_KEY": "nope"}
    )
    assert "OPENAI_API_KEY" in applied
    assert os.environ.get("OPENAI_API_KEY") == "sk-test"
    assert "EVIL_KEY" not in os.environ


def test_run_config_ignores_client_interrupt_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    from symvion import Symvion, TenantConfig

    runtime = Symvion(
        TenantConfig(
            tenant_id="t1",
            enabled_agents=[],
            logging={"enabled": False},
            trust_hitl_edit=False,
        )
    )
    cfg = runtime._run_config(
        "t1",
        "sess-1",
        interrupt_before_tools=["safe_form"],
        extra_metadata={
            "interrupt_before_tools": ["submit_claim"],
            "trust_hitl_edit": True,
            "tenant_id": "other",
        },
    )
    meta = cfg["metadata"]
    assert meta["interrupt_before_tools"] == ["safe_form"]
    assert meta["tenant_id"] == "t1"
    assert meta["session_id"] == "sess-1"
    assert meta["trust_hitl_edit"] is False


def test_claims_tools_allowlist_rejects_unknown_host():
    from symvion.tools.claims_tools import ClaimsTools

    with pytest.raises(ValueError):
        ClaimsTools(
            "t1",
            backend_url="https://evil.example",
            allowed_outbound_hosts=["api.example.com"],
        )
