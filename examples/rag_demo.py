"""
examples/rag_demo.py
~~~~~~~~~~~~~~~~~~~~
Demonstrates how to wire the Symvion RAG system end-to-end.

This file shows the *integration pattern* only — it contains no
business-specific prompts, index names, or company logic.

Run (from the ai-runtime directory):
    python examples/rag_demo.py

Requirements:
    - GOOGLE_CLOUD_PROJECT env var set (or pass project_id directly)
    - Application Default Credentials configured (gcloud auth login)
    - Optional: VERTEX_DATA_STORE_ID env var for a live data store.
      Without it the retriever runs in stub mode (mock results).
"""

from __future__ import annotations

import asyncio
import os

# ---------------------------------------------------------------------------
# 1. Import from Symvion — the public API surface
# ---------------------------------------------------------------------------
from symvion import (
    Symvion,
    TenantConfig,
    BaseRetriever,        # noqa: F401  (shown for clarity)
    RAGPipeline,
    RetrievalTool,
    VertexRetriever,
)


# ---------------------------------------------------------------------------
# 2. Configure the runtime
#    All business config lives here, OUTSIDE Symvion.
# ---------------------------------------------------------------------------

config = TenantConfig(
    tenant_id="demo-tenant",
    llm_provider="mock",          # swap for "google_genai", "openai", etc.
    llm_config={"model": "mock"},
    logging={"enabled": True, "level": "INFO", "show_json": True},
)

runtime = Symvion(config)


# ---------------------------------------------------------------------------
# 3. Create a retriever
#    Pass data_store_id=None → stub mode (no live GCP project needed).
#    Pass a real data_store_id for live Vertex AI Search.
# ---------------------------------------------------------------------------

retriever = VertexRetriever(
    project_id=os.getenv("GOOGLE_CLOUD_PROJECT", "demo-project"),
    location="global",
    data_store_id=os.getenv("VERTEX_DATA_STORE_ID"),   # None = stub mode
    default_top_k=5,
)


# ---------------------------------------------------------------------------
# 4. Register the retriever and tool with the runtime
# ---------------------------------------------------------------------------

runtime.set_retriever("demo-tenant", retriever)

runtime.register_tool(
    tenant="demo-tenant",
    name="retrieve_knowledge",
    tool=RetrievalTool(retriever, default_top_k=5),
)


# ---------------------------------------------------------------------------
# 5. Direct tool invocation (agent-style)
# ---------------------------------------------------------------------------

async def demo_tool_call() -> None:
    print("\n--- RetrievalTool demo ---")
    result = await runtime.tools.execute(
        "retrieve_knowledge",
        {"query": "What is the claims process?", "top_k": 3},
    )
    print(f"Query:     {result['query']}")
    print(f"Docs:      {result['doc_count']}")
    print(f"Fallback:  {result['fallback']}")
    for i, doc in enumerate(result["documents"], 1):
        print(f"\n  [{i}] {doc['text'][:120]}")
        print(f"      metadata={doc['metadata']}")


# ---------------------------------------------------------------------------
# 6. Full RAG pipeline (retrieval → prompt → generation)
# ---------------------------------------------------------------------------

async def demo_pipeline() -> None:
    print("\n--- RAGPipeline demo ---")
    llm = runtime.config  # will be replaced by a real LLM in production

    # Use the LLM from the provider factory so this stays provider-agnostic
    from symvion.providers.factory import ProviderFactory
    llm = ProviderFactory.get_provider(
        config.llm_provider,
        config.llm_config,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=3,
        fallback_response="Sorry, I could not find relevant information.",
    )

    output = await pipeline.run("Explain the policy renewal procedure.")
    print(f"Query:      {output['query']}")
    print(f"Duration:   {output['duration_s']}s")
    print(f"Documents:  {len(output['documents'])}")
    print(f"\nAnswer:\n{output['answer']}")


# ---------------------------------------------------------------------------
# 7. Custom retriever (shows extensibility — business logic stays outside)
# ---------------------------------------------------------------------------

class MyCustomRetriever(BaseRetriever):
    """
    Example of a custom retriever — replace the body with your own
    backend (OpenSearch, Pinecone, etc.).
    """

    async def search(self, query: str, *, top_k: int = 5, **kwargs):
        # Simulates a call to any custom retrieval backend
        return [
            {
                "text": f"Custom result for '{query}' (doc {i + 1})",
                "metadata": {"source": "custom-backend", "rank": i + 1},
            }
            for i in range(top_k)
        ]


async def demo_custom_retriever() -> None:
    print("\n--- Custom retriever demo ---")
    custom = MyCustomRetriever()
    runtime.set_retriever("demo-tenant-custom", custom)
    runtime.register_tool(
        tenant="demo-tenant-custom",
        name="custom_retrieve",
        tool=RetrievalTool(custom, default_top_k=3),
    )
    result = await runtime.tools.execute(
        "custom_retrieve",
        {"query": "How do I file a claim?"},
    )
    print(f"Retrieved {result['doc_count']} documents:")
    for doc in result["documents"]:
        print(f"  • {doc['text']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    await demo_tool_call()
    await demo_pipeline()
    await demo_custom_retriever()

    print("\n✅ RAG demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
