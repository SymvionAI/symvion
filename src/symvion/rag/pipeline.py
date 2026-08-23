"""
symvion.rag.pipeline
~~~~~~~~~~~~~~~~~~~~
High-level RAG pipeline that wires retrieval and generation together.

RAGPipeline is intentionally thin:
- It is provider-agnostic (accepts any BaseRetriever + any async LLM).
- It contains NO business logic, prompts, or domain knowledge.
- Callers are responsible for injecting a prompt template if they need
  something other than the sensible default.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from symvion.rag.base import BaseRetriever
from symvion.core.logger import logger
from symvion.core.exceptions import SymvionError


# ---------------------------------------------------------------------------
# Default prompt builder (fully generic — override at call site)
# ---------------------------------------------------------------------------

def _default_prompt_builder(query: str, documents: List[Dict[str, Any]]) -> str:
    """Build a plain RAG prompt from retrieved documents and the user query."""
    if not documents:
        return f"Answer the following question as best you can:\n\n{query}"

    context_text = "\n\n".join(
        f"[Document {i + 1}]\n{doc['text']}" for i, doc in enumerate(documents)
    )
    return (
        "Use the following retrieved context to answer the question.\n"
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates retrieval → prompt-building → generation.

    Args:
        retriever:      Any BaseRetriever implementation.
        llm:            Any async LangChain BaseChatModel (or object with an
                        ``ainvoke`` / ``generate`` coroutine that accepts a
                        string and returns a string).
        prompt_builder: Optional callable ``(query, documents) -> str``.
                        Defaults to a generic RAG template.  Inject your own
                        domain-specific builder from outside Symvion.
        top_k:          Default number of documents to retrieve per query.
        fallback_response: String returned when retrieval yields no results
                        and ``llm`` is also unavailable. Defaults to a neutral
                        message.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Any,
        *,
        prompt_builder: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
        top_k: int = 5,
        fallback_response: str = (
            "I was unable to find relevant information to answer your question."
        ),
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder or _default_prompt_builder
        self.top_k = top_k
        self.fallback_response = fallback_response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a single query.

        Args:
            query:            The user's natural-language question.
            top_k:            Overrides instance-level ``top_k`` for this call.
            retriever_kwargs: Extra kwargs forwarded to ``retriever.search()``.

        Returns:
            A dict with keys:
            - ``"answer"``    — Generated text from the LLM.
            - ``"documents"`` — Raw documents returned by the retriever.
            - ``"query"``     — The original query (for traceability).
            - ``"duration_s"``— Wall-clock seconds for the full pipeline run.
        """
        k = top_k if top_k is not None else self.top_k
        extra = retriever_kwargs or {}

        start = time.perf_counter()

        # 1. Retrieve
        logger.info("RAG_PIPELINE_RETRIEVAL_START", None, query=query, top_k=k)
        documents = await self.retriever.search(query, top_k=k, **extra)
        logger.info(
            "RAG_PIPELINE_RETRIEVAL_END",
            None,
            query=query,
            doc_count=len(documents),
        )

        # 2. Handle no-results fallback
        if not documents:
            logger.warning("RAG_PIPELINE_NO_RESULTS", None, query=query)
            return {
                "answer": self.fallback_response,
                "documents": [],
                "query": query,
                "duration_s": round(time.perf_counter() - start, 4),
            }

        # 3. Build prompt
        prompt = self.prompt_builder(query, documents)

        # 4. Generate
        logger.info("RAG_PIPELINE_GENERATION_START", None, query=query)
        answer = await self._generate(prompt)
        logger.info("RAG_PIPELINE_GENERATION_END", None, query=query)

        duration = round(time.perf_counter() - start, 4)
        logger.info("RAG_PIPELINE_COMPLETE", None, query=query, duration_s=duration)

        return {
            "answer": answer,
            "documents": documents,
            "query": query,
            "duration_s": duration,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate(self, prompt: str) -> str:
        """
        Invoke the LLM using whichever async interface it exposes.

        Supports:
        - LangChain BaseChatModel  → ``ainvoke(messages)``
        - Any object with          → ``generate(prompt)`` coroutine
        """
        from langchain_core.messages import HumanMessage

        try:
            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                return response.content if hasattr(response, "content") else str(response)

            if hasattr(self.llm, "generate"):
                return await self.llm.generate(prompt)

            raise SymvionError(
                f"LLM object {type(self.llm).__name__!r} has neither "
                "'ainvoke' nor 'generate'. Cannot generate a response."
            )
        except SymvionError:
            raise
        except Exception as exc:
            raise SymvionError(f"LLM generation failed: {exc}") from exc
