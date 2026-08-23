"""
symvion.rag.base
~~~~~~~~~~~~~~~~
Abstract base interface for all retrievers in the Symvion RAG system.

Implementing a new retriever is as simple as subclassing BaseRetriever
and providing an async ``search`` method. No framework internals need
to change — register the retriever with the runtime and it's ready.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRetriever(ABC):
    """
    Abstract retriever interface for the Symvion RAG module.

    All retrieval backends (vector databases, search APIs, etc.) must
    implement this interface so that the rest of the framework can
    remain provider-agnostic.

    Returned documents must follow the standard schema::

        [
            {
                "text": "The retrieved chunk of text.",
                "metadata": {"source": "...", "score": 0.92, ...},
            },
            ...
        ]

    The ``metadata`` dict is open-ended; providers may include fields
    such as ``score``, ``source``, ``chunk_id``, etc.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic/lexical search and return ranked documents.

        Args:
            query:  The natural-language query to search for.
            top_k:  Maximum number of documents to return (default 5).
            **kwargs: Provider-specific parameters (filters, namespaces, etc.).

        Returns:
            A list of document dicts, each containing at minimum:
            ``{"text": str, "metadata": dict}``.
            Returns an empty list when nothing is found — callers should
            handle the no-results case gracefully.

        Raises:
            RetrieverError: If the underlying retrieval call fails.
        """
        ...

    # ------------------------------------------------------------------
    # Optional lifecycle hooks (override as needed)
    # ------------------------------------------------------------------

    async def on_search_start(self, query: str, top_k: int) -> None:
        """Hook called immediately before each search. No-op by default."""

    async def on_search_end(
        self,
        query: str,
        top_k: int,
        results: List[Dict[str, Any]],
    ) -> None:
        """Hook called immediately after each search. No-op by default."""
