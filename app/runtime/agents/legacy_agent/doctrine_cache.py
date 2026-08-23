"""
Load and cache founder doctrine principles from configuration.
Single load on first access; in-memory reuse for performance.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Resolve config path relative to this package: ai-runtime/app/runtime/agents/legacy_agent
# -> ai-runtime/config/doctrine/principles.yaml
_THIS_DIR = Path(__file__).resolve().parent
_RUNTIME = _THIS_DIR.parent  # agents
_APP = _RUNTIME.parent  # runtime
_APP_ROOT = _APP.parent  # app
_AI_RUNTIME_ROOT = _APP_ROOT.parent  # ai-runtime
_DEFAULT_DOCTRINE_PATH = _AI_RUNTIME_ROOT / "config" / "doctrine" / "principles.yaml"

_cached_principles: List[Dict[str, Any]] | None = None


def get_principles(doctrine_path: Path | str | None = None) -> List[Dict[str, Any]]:
    """
    Load founder doctrine principles from YAML and cache in memory.
    Subsequent calls return the cached list; no per-request file I/O.

    Args:
        doctrine_path: Optional path to principles YAML. Defaults to
            ai-runtime/config/doctrine/principles.yaml.

    Returns:
        List of principle dicts, each with 'name' and 'rules' (list of strings).
    """
    global _cached_principles
    if _cached_principles is not None:
        return _cached_principles

    path = Path(doctrine_path) if doctrine_path else _DEFAULT_DOCTRINE_PATH
    if not path.is_absolute():
        path = _AI_RUNTIME_ROOT / path

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; doctrine principles will be empty")
        _cached_principles = []
        return _cached_principles

    if not path.exists():
        logger.warning("Doctrine principles file not found: %s", path)
        _cached_principles = []
        return _cached_principles

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("principles") or []
        # Normalize to list of dicts with name and rules
        out: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                out.append({
                    "name": item["name"],
                    "rules": item.get("rules") or [],
                })
            else:
                logger.debug("Skipping invalid principle entry: %s", item)
        _cached_principles = out
        logger.info("Loaded %d doctrine principles from %s", len(_cached_principles), path)
        return _cached_principles
    except Exception as e:
        logger.exception("Failed to load doctrine principles from %s: %s", path, e)
        _cached_principles = []
        return _cached_principles


def clear_cache() -> None:
    """Clear the in-memory principles cache (for tests or config reload)."""
    global _cached_principles
    _cached_principles = None
