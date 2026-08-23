"""
symvion.tools.retrieval
~~~~~~~~~~~~~~~~~~~~~~~~
RetrievalTool — a Symvion tool that wraps any BaseRetriever.

Design goals:
  ✅  No business logic
  ✅  No provider-specific assumptions
  ✅  Compatible with ToolRegistry.register_tool()
  ✅  Works with context.tools.execute("retrieve_knowledge", {...})
  ✅  Supports top_k override per call
  ✅  Logs retrieval calls for observability
  ✅  Returns a structured fallback when no documents are found
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from symvion.rag.base import BaseRetriever
from symvion.core.context import TenantContext
from symvion.core.logger import logger


_FALLBACK_DOCUMENTS: List[Dict[str, Any]] = []


class RetrievalTool:
    """
    A generic Symvion tool that executes a retrieval query via any
    :class:`~symvion.rag.base.BaseRetriever` implementation.

    Register it once per tenant::

        runtime.register_tool(
            tenant="acme",
            name="retrieve_knowledge",
            tool=RetrievalTool(retriever),
        )

    Agents call it through the tool registry::

        result = await context.tools.execute(
            "retrieve_knowledge",
            {"query": user_query, "top_k": 10},
            context,
        )

    Args:
        retriever:          Any BaseRetriever implementation.
        default_top_k:      Default number of documents if ``top_k`` is not
                            supplied in ``input_data``.
        fallback_message:   Human-readable text placed in ``documents`` when
                            retrieval returns nothing (empty list).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        default_top_k: int = 5,
        fallback_message: str = "No relevant documents were found for your query.",
    ) -> None:
        self.retriever = retriever
        self.default_top_k = default_top_k
        self.fallback_message = fallback_message

    # ------------------------------------------------------------------
    # Public API — matches the contract expected by ToolRegistry
    # ------------------------------------------------------------------

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute a retrieval query.

        Args:
            input_data: Must contain ``"query"`` (str).  Optionally accepts:
                - ``"top_k"`` (int) — overrides instance default.
                - Any extra key is forwarded to ``retriever.search(**kwargs)``.
            context:    Optional TenantContext for structured logging.

        Returns:
            ``{"documents": List[Dict], "query": str, "doc_count": int,
               "fallback": bool, "duration_s": float}``

        Raises:
            ValueError: If ``"query"`` is missing or empty.
        """
        query: str = input_data.get("query", "").strip()
        if not query:
            raise ValueError(
                "RetrievalTool requires a non-empty 'query' in input_data."
            )

        top_k: int = int(input_data.get("top_k", self.default_top_k))

        # Forward any extra keys to the retriever as kwargs
        _reserved = {"query", "top_k"}
        extra_kwargs: Dict[str, Any] = {
            k: v for k, v in input_data.items() if k not in _reserved
        }

        logger.info(
            "RETRIEVAL_TOOL_CALLED",
            context,
            query=query,
            top_k=top_k,
            retriever=type(self.retriever).__name__,
        )

        start = time.perf_counter()

        try:
            documents = await self.retriever.search(
                query, top_k=top_k, **extra_kwargs
            )
        except Exception as exc:
            logger.error(
                "RETRIEVAL_TOOL_FAILED",
                context,
                query=query,
                error=str(exc),
            )
            raise

        duration = round(time.perf_counter() - start, 4)
        fallback = False

        if not documents:
            logger.warning(
                "RETRIEVAL_TOOL_NO_RESULTS",
                context,
                query=query,
                top_k=top_k,
            )
            documents = [
                {
                    "text": self.fallback_message,
                    "metadata": {"fallback": True},
                }
            ]
            fallback = True

        logger.info(
            "RETRIEVAL_TOOL_SUCCESS",
            context,
            query=query,
            doc_count=len(documents),
            fallback=fallback,
            duration_s=duration,
        )

        return {
            "documents": documents,
            "query": query,
            "doc_count": len(documents),
            "fallback": fallback,
            "duration_s": duration,
        }

    # ------------------------------------------------------------------
    # Convenience: make the tool directly callable (useful in tests)
    # ------------------------------------------------------------------

    async def __call__(
        self,
        input_data: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> Dict[str, Any]:
        return await self.execute(input_data, context)
