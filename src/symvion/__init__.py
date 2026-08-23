"""Multi-tenant AI orchestration framework powered by LangGraph."""

from importlib import import_module
from typing import Any

__all__ = [
    "Symvion",
    "TenantConfig",
    "AgentRegistration",
    "ConfigRegistry",
    "tool",
    "BaseRetriever",
    "RAGPipeline",
    "RetrievalTool",
    "VertexRetriever",
]

_EXPORTS = {
    "Symvion": "symvion.core.runtime",
    "TenantConfig": "symvion.config.models",
    "AgentRegistration": "symvion.config.models",
    "ConfigRegistry": "symvion.config.loader",
    "BaseRetriever": "symvion.rag.base",
    "RAGPipeline": "symvion.rag.pipeline",
    "RetrievalTool": "symvion.tools.retrieval",
    "VertexRetriever": "symvion.rag.providers.vertex",
    "tool": "langchain_core.tools",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
