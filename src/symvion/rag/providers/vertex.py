"""
symvion.rag.providers.vertex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vertex AI Search adapter for the Symvion RAG module.

This file is a *framework adapter only*.  It isolates all Vertex AI SDK
calls behind the standard BaseRetriever interface and contains:

  ✅  Provider SDK initialisation
  ✅  Query → Vertex AI Search → standard document format mapping
  ✅  top_k / num_results support
  ✅  Structured logging via the Symvion logger

  ❌  No index/datastore names (pass them in at construction time)
  ❌  No company-specific logic, prompts, or domain rules
  ❌  No authentication secrets (use ADC or service-account env vars)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from symvion.rag.base import BaseRetriever
from symvion.core.logger import logger
from symvion.core.exceptions import SymvionError


class RetrieverError(SymvionError):
    """Raised when a retrieval backend call fails."""


class VertexRetriever(BaseRetriever):
    """
    Vertex AI Search retriever adapter.

    Connects to a *caller-supplied* Vertex AI Search data store and returns
    results in the standard Symvion document format.  All tenant- and
    domain-specific configuration (data store IDs, serving configs, etc.)
    must be injected at construction time — nothing is hardcoded here.

    Args:
        project_id:      GCP project ID.
        location:        Vertex AI region (default ``"global"`` for
                         Vertex AI Search; use ``"us-central1"`` for
                         Vertex AI Matching Engine / Vector Search).
        data_store_id:   The Vertex AI Search data store to query.
                         Leave ``None`` to use a placeholder stub that
                         returns mock results (useful for local development
                         without a live GCP project).
        serving_config:  Serving config resource name, e.g.
                         ``"default_config"``.  Required if
                         ``data_store_id`` is provided.
        default_top_k:   Default number of results returned per query.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "global",
        *,
        data_store_id: Optional[str] = None,
        serving_config: str = "default_config",
        default_top_k: int = 5,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.data_store_id = data_store_id
        self.serving_config = serving_config
        self.default_top_k = default_top_k

        # Lazy SDK client (created on first search call)
        self._client: Any = None

        logger.info(
            "VERTEX_RETRIEVER_INIT",
            None,
            project_id=project_id,
            location=location,
            data_store_id=data_store_id or "stub-mode",
        )

    # ------------------------------------------------------------------
    # BaseRetriever implementation
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Search the configured Vertex AI data store.

        When ``data_store_id`` is ``None`` (stub mode), returns a single
        placeholder document so the pipeline can be exercised without a
        live GCP project.

        Args:
            query:  Natural-language search query.
            top_k:  Maximum documents to return.
            **kwargs: Forwarded to the Vertex AI SDK (e.g., ``filter``).

        Returns:
            List of ``{"text": str, "metadata": dict}`` dicts.
        """
        k = top_k or self.default_top_k

        await self.on_search_start(query, k)

        start = time.perf_counter()
        logger.info(
            "VERTEX_RETRIEVER_SEARCH_START",
            None,
            query=query,
            top_k=k,
            project_id=self.project_id,
            data_store_id=self.data_store_id or "stub",
        )

        # ------------------------------------------------------------------
        # Stub mode — no live data store configured
        # ------------------------------------------------------------------
        if self.data_store_id is None:
            logger.warning(
                "VERTEX_RETRIEVER_STUB_MODE",
                None,
                reason="data_store_id not provided",
            )
            results = [
                {
                    "text": f"[STUB] Sample result for query: {query!r}",
                    "metadata": {
                        "source": "stub",
                        "score": 1.0,
                        "project_id": self.project_id,
                    },
                }
            ]
            await self.on_search_end(query, k, results)
            return results

        # ------------------------------------------------------------------
        # Live mode — delegate to Vertex AI Search SDK
        # ------------------------------------------------------------------
        try:
            results = await self._live_search(query, k, **kwargs)
        except Exception as exc:
            logger.error(
                "VERTEX_RETRIEVER_SEARCH_FAILED",
                None,
                query=query,
                error=str(exc),
            )
            raise RetrieverError(
                f"Vertex AI search failed for query {query!r}: {exc}"
            ) from exc

        duration = round(time.perf_counter() - start, 4)
        logger.info(
            "VERTEX_RETRIEVER_SEARCH_END",
            None,
            query=query,
            doc_count=len(results),
            duration_s=duration,
        )

        await self.on_search_end(query, k, results)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        """Lazily initialise the Vertex AI Search client."""
        if self._client is not None:
            return self._client

        try:
            from google.cloud import discoveryengine_v1 as discoveryengine  # noqa: F401
            import asyncio

            # Use asyncio executor for the sync SDK call
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None,
                lambda: discoveryengine.SearchServiceClient(),
            )
            logger.info("VERTEX_RETRIEVER_CLIENT_READY", None, project_id=self.project_id)
        except ImportError as exc:
            raise RetrieverError(
                "google-cloud-discoveryengine is not installed. "
                "Run: pip install google-cloud-discoveryengine"
            ) from exc

        return self._client

    async def _live_search(
        self, query: str, top_k: int, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Execute the search against Vertex AI Search and normalise results."""
        import asyncio
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = await self._get_client()

        serving_config_path = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/collections/default_collection/dataStores/{self.data_store_id}"
            f"/servingConfigs/{self.serving_config}"
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config_path,
            query=query,
            page_size=top_k,
            **{k: v for k, v in kwargs.items() if hasattr(discoveryengine.SearchRequest, k)},
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.search(request))

        return [
            self._normalise_result(result)
            for result in list(response.results)[:top_k]
        ]

    @staticmethod
    def _normalise_result(result: Any) -> Dict[str, Any]:
        """
        Map a Vertex AI SearchResult proto to the standard Symvion format.

        ``{"text": str, "metadata": dict}``
        """
        doc = result.document

        # Extract extractive answers / segments if present
        text_parts: List[str] = []
        metadata: Dict[str, Any] = {"doc_id": doc.id if hasattr(doc, "id") else ""}

        derived = getattr(doc, "derived_struct_data", None)
        if derived:
            # Extractive answers
            for answer in derived.get("extractive_answers", []):
                if "content" in answer:
                    text_parts.append(answer["content"])

            # Extractive segments
            for segment in derived.get("extractive_segments", []):
                if "content" in segment:
                    text_parts.append(segment["content"])

            metadata.update(
                {
                    k: v
                    for k, v in derived.items()
                    if k not in ("extractive_answers", "extractive_segments")
                }
            )

        # Fall back to struct data
        if not text_parts:
            struct = getattr(doc, "struct_data", None)
            if struct:
                content = struct.get("content") or struct.get("text") or str(struct)
                text_parts.append(content)

        # Final fallback
        if not text_parts:
            text_parts.append(f"[No extractable text for doc {metadata['doc_id']}]")

        return {
            "text": "\n".join(text_parts),
            "metadata": metadata,
        }
