"""Security helpers for outbound URLs, path sandboxing, and secret handling."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

# Credential-related env keys that may be set when unset (never overwrite).
ALLOWED_PROVIDER_ENV_KEYS: Set[str] = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
}

CLIENT_ERROR_MESSAGE = "An internal error occurred while processing the request."

# Client metadata keys that must never override host-owned run config.
UNTRUSTED_METADATA_KEYS = (
    "interrupt_before_tools",
    "trust_hitl_edit",
    "iam_policies",
    "tenant_id",
)

# Hostnames that are never valid outbound targets (cloud metadata / cluster DNS).
BLOCKED_OUTBOUND_HOSTS = {
    "metadata.google.internal",
    "metadata.google.com",
    "instance-data",
    "kubernetes.default",
    "kubernetes.default.svc",
    "kubernetes.default.svc.cluster.local",
}


def hash_secret(value: Optional[str], *, prefix: str = "") -> str:
    """Return a stable non-reversible fingerprint for cache keys / logs."""
    if not value:
        return f"{prefix}none" if prefix else "none"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def redact_secret(value: Optional[str]) -> str:
    """Safe log representation of a secret."""
    if not value:
        return "absent"
    return f"present:{hash_secret(value)}"


def client_safe_error(exc: BaseException) -> str:
    """Generic error string safe to return to API clients."""
    return CLIENT_ERROR_MESSAGE


def _unwrap_ip(addr: ipaddress._BaseAddress) -> ipaddress._BaseAddress:
    """Map IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) to the inner IPv4 address."""
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _ip_is_blocked(addr: ipaddress._BaseAddress) -> bool:
    ip = _unwrap_ip(addr)
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _parse_literal_ip(host: str) -> Optional[ipaddress._BaseAddress]:
    """Parse a hostname that is already an IP, including decimal IPv4 (2130706433)."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if host.isdigit():
        try:
            return ipaddress.IPv4Address(int(host))
        except (ValueError, OverflowError):
            return None
    return None


def _is_private_or_local_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return True
    if host in BLOCKED_OUTBOUND_HOSTS:
        return True
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return True
    literal = _parse_literal_ip(host)
    if literal is not None:
        return _ip_is_blocked(literal)
    try:
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if _ip_is_blocked(ip):
                return True
    except (socket.gaierror, OSError, ValueError):
        # If we cannot resolve, treat as unsafe (fail closed).
        return True
    return False


def validate_outbound_url(
    url: str,
    *,
    allowed_hosts: Optional[List[str]] = None,
    allow_localhost: bool = True,
    require_https: bool = False,
) -> str:
    """
    Validate an outbound HTTP(S) URL.

    Raises ValueError when the URL is unsafe (bad scheme, private host without
    allow_localhost, or host not on allowlist when configured).
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL is required")

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed")
    if require_https and scheme != "https":
        raise ValueError("HTTPS is required for outbound URLs")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    host = parsed.hostname.lower()
    is_local = host in {"localhost", "127.0.0.1", "::1"}

    if allowed_hosts:
        allowed = {h.lower() for h in allowed_hosts}
        if host not in allowed and not (allow_localhost and is_local):
            raise ValueError(f"Host '{host}' is not in the outbound allowlist")

    if _is_private_or_local_host(host):
        if not (allow_localhost and is_local):
            raise ValueError("Outbound requests to private/link-local addresses are blocked")

    return url.rstrip("/")


def resolve_sandboxed_path(
    file_path: str,
    sandbox_dir: Optional[str] = None,
) -> Path:
    """
    Resolve ``file_path`` under a sandbox directory.

    Defaults to ``SYMVION_FILE_SANDBOX`` or the process temp dir.
    Rejects path traversal outside the sandbox.
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path is required")

    root = sandbox_dir or os.environ.get("SYMVION_FILE_SANDBOX")
    if not root:
        import tempfile

        root = tempfile.gettempdir()

    sandbox = Path(root).resolve()
    candidate = Path(file_path)
    # Absolute paths must still land inside the sandbox.
    resolved = (sandbox / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    try:
        resolved.relative_to(sandbox)
    except ValueError as exc:
        raise ValueError("file_path escapes the configured sandbox") from exc

    if not resolved.is_file():
        raise ValueError("file_path must refer to an existing file inside the sandbox")

    return resolved


def sanitize_request_metadata(
    metadata: Optional[Dict[str, Any]],
    *,
    trust_client_role: bool = False,
    trust_client_agent_id: bool = False,
    default_user_role: str = "user",
) -> Dict[str, Any]:
    """
    Strip untrusted client security fields unless explicitly enabled by the host.
    """
    meta = dict(metadata or {})
    for key in UNTRUSTED_METADATA_KEYS:
        meta.pop(key, None)
    if not trust_client_role:
        meta.pop("user_role", None)
        meta["user_role"] = default_user_role
    elif "user_role" not in meta:
        meta["user_role"] = default_user_role

    if not trust_client_agent_id:
        meta.pop("agent_id", None)

    return meta


def apply_provider_env_vars(env_vars: Optional[Dict[str, str]]) -> List[str]:
    """
    Apply only known provider credential keys when not already set.

    Returns the list of keys that were set. Arbitrary env mutation is refused
    to avoid cross-tenant credential bleed in shared processes.
    """
    applied: List[str] = []
    if not env_vars:
        return applied
    for key, value in env_vars.items():
        if key not in ALLOWED_PROVIDER_ENV_KEYS:
            continue
        if key in os.environ and os.environ.get(key):
            continue
        if value is None:
            continue
        os.environ[key] = str(value)
        applied.append(key)
    return applied
