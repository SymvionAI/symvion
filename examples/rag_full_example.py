"""
examples/rag_full_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full end-to-end example: Symvion runtime + RAGPipeline.

Covers:
  1. Building the runtime (TenantConfig + Symvion)
  2. Creating and registering a retriever
  3. Registering a RetrievalTool for agent-style calls
  4. Using RAGPipeline for standalone retrieval + generation
  5. Using context.tools.execute() the way an agent would

Run from the ai-runtime directory:
    PYTHONPATH=src venv/bin/python3 examples/rag_full_example.py
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Symvion imports — the full public API surface
# ---------------------------------------------------------------------------
from symvion import (
    Symvion,
    TenantConfig,
    BaseRetriever,
    RAGPipeline,
    RetrievalTool,
    VertexRetriever,
)
from symvion.providers.factory import ProviderFactory
from symvion.core.context import TenantContext


# ===========================================================================
# PART 1 — Build the runtime
# ===========================================================================
# TenantConfig is where ALL business/deployment config lives.
# Nothing below this block is tenant-specific.

config = TenantConfig(
    tenant_id="acme",
    llm_provider="mock",             # swap: "google_genai", "openai", "anthropic"
    llm_config={"model": "mock"},    # swap: {"model": "gemini-1.5-pro", "temperature": 0.2}
    logging={
        "enabled": True,
        "level": "INFO",
        "show_json": False,          # plain-text logs for readability in this demo
    },
)

runtime = Symvion(config)

# What Symvion(config) gives you:
#   runtime.tools       → ToolRegistry  (register + invoke tools)
#   runtime._retrievers → {}            (populated below)
#   runtime.agents      → AgentRegistry (unchanged — not used here)
#   runtime.memory_store                (unchanged — not used here)


# ===========================================================================
# PART 2 — Create and register a retriever
# ===========================================================================
# VertexRetriever with data_store_id=None → stub mode.
# Swap data_store_id for a real Vertex AI Search data store in production.

retriever = VertexRetriever(
    project_id="my-gcp-project",
    location="global",
    data_store_id=None,          # None = stub mode (no live GCP needed)
    default_top_k=5,
)

# Attach the retriever to the runtime under the "acme" tenant.
# You can register different retrievers for different tenants:
#   runtime.set_retriever("acme",    vertex_retriever)
#   runtime.set_retriever("hr-corp", opensearch_retriever)
runtime.set_retriever("acme", retriever)


# ===========================================================================
# PART 3 — Register a RetrievalTool for agent-style invocation
# ===========================================================================
# Agents inside the graph call tools by name through context.tools.execute().
# Register the tool once at startup — agents never import or construct it.

runtime.register_tool(
    tenant="acme",
    name="retrieve_knowledge",
    tool=RetrievalTool(
        retriever,
        default_top_k=5,
        fallback_message="No relevant documents were found for your query.",
    ),
)

# You can register as many named tools as you need:
#   runtime.register_tool("acme", "search_products",  RetrievalTool(product_retriever))
#   runtime.register_tool("acme", "search_policies",  RetrievalTool(policy_retriever))


# ===========================================================================
# PART 4 — RAGPipeline: retrieval + generation in one call
# ===========================================================================
# Use RAGPipeline in scripts, API endpoints, or batch jobs —
# anywhere OUTSIDE the LangGraph agent loop.

async def demo_pipeline() -> None:
    print("\n" + "=" * 60)
    print("DEMO: RAGPipeline (retrieval + generation)")
    print("=" * 60)

    # Get the LLM from the same provider factory the runtime uses
    # so the pipeline stays consistent with the rest of the system.
    llm = ProviderFactory.get_provider(config.llm_provider, config.llm_config)

    pipeline = RAGPipeline(
        retriever=retriever,           # the same retriever registered above
        llm=llm,
        top_k=3,
        fallback_response="I could not find enough information to answer that.",

        # Optional: inject your own prompt builder.
        # Keep domain-specific prompts HERE, outside Symvion.
        # prompt_builder=my_insurance_prompt_builder,
    )

    # pipeline.run() does three things in sequence:
    #   1. retriever.search(query, top_k=3)   → documents
    #   2. prompt_builder(query, documents)    → prompt string
    #   3. llm.ainvoke([HumanMessage(prompt)]) → answer
    output = await pipeline.run("What are the steps to renew a policy?")

    print(f"Query     : {output['query']}")
    print(f"Duration  : {output['duration_s']}s")
    print(f"Documents : {output['doc_count'] if 'doc_count' in output else len(output['documents'])}")
    print(f"Answer    :\n  {output['answer']}")

    print("\nRetrieved documents:")
    for i, doc in enumerate(output["documents"], 1):
        print(f"  [{i}] {doc['text'][:80]}")
        print(f"       metadata={doc['metadata']}")


# ===========================================================================
# PART 5 — Agent-style invocation via context.tools.execute()
# ===========================================================================
# This is exactly what a BaseAgent.execute() method does internally.
# The agent never constructs a retriever — it just calls the tool by name.

async def demo_agent_style_call() -> None:
    print("\n" + "=" * 60)
    print("DEMO: Agent-style tool call via context.tools.execute()")
    print("=" * 60)

    # In production, TenantContext is created by the runtime per-request.
    # We create it manually here only to demonstrate the pattern.
    ctx = TenantContext(
        tenant_id="acme",
        metadata={"user_role": "agent", "session_id": "demo-session"},
    )

    user_query = "How do I file a claim?"

    # This is the exact call an agent makes inside execute():
    rag_result = await runtime.tools.execute(
        "retrieve_knowledge",               # tool name
        {"query": user_query, "top_k": 3},  # input_data
        ctx,                                # tenant context (for logging + IAM)
    )

    print(f"Query      : {rag_result['query']}")
    print(f"Doc count  : {rag_result['doc_count']}")
    print(f"Fallback   : {rag_result['fallback']}")
    print(f"Duration   : {rag_result['duration_s']}s")

    # The agent would now build its own prompt and call self.llm:
    docs_text = "\n".join(
        f"[{i+1}] {doc['text']}" for i, doc in enumerate(rag_result["documents"])
    )
    print(f"\nContext the agent would send to its LLM:\n{docs_text}")


# ===========================================================================
# PART 6 — Custom retriever (shows how easy it is to add a new provider)
# ===========================================================================
# Subclass BaseRetriever, implement search(), register it — done.
# Zero changes to Symvion framework code.

class InMemoryRetriever(BaseRetriever):
    """Toy in-memory retriever for local development and testing."""

    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        self._documents = documents

    async def search(
        self, query: str, *, top_k: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        # In production: embed query, cosine-similarity search, etc.
        # Here: return the first top_k docs as a simple demo.
        results = self._documents[:top_k]
        return [
            {
                "text": doc["text"],
                "metadata": {**doc.get("metadata", {}), "matched_query": query},
            }
            for doc in results
        ]


async def demo_custom_retriever() -> None:
    print("\n" + "=" * 60)
    print("DEMO: Custom InMemoryRetriever")
    print("=" * 60)

    custom_retriever = InMemoryRetriever(
        documents=[
            {"text": "Policy renewal requires 30 days notice.", "metadata": {"source": "handbook"}},
            {"text": "Claims must be filed within 90 days of the incident.", "metadata": {"source": "handbook"}},
            {"text": "Premium payments are due on the 1st of each month.", "metadata": {"source": "faq"}},
        ]
    )

    # Register under a separate tenant to avoid collisions
    runtime.set_retriever("acme-test", custom_retriever)
    runtime.register_tool(
        tenant="acme-test",
        name="retrieve_handbook",
        tool=RetrievalTool(custom_retriever, default_top_k=2),
    )

    result = await runtime.tools.execute(
        "retrieve_handbook",
        {"query": "When do I need to renew?", "top_k": 2},
    )

    print(f"Retrieved {result['doc_count']} docs:")
    for doc in result["documents"]:
        print(f"  • {doc['text']}")
        print(f"    {doc['metadata']}")

    # Inspect all registered tools across both tenants
    print(f"\nAll registered tools: {runtime.tools.all_tools}")


# ===========================================================================
# Entry point
# ===========================================================================

async def main() -> None:
    await demo_pipeline()
    await demo_agent_style_call()
    await demo_custom_retriever()
    print("\n✅  All demos complete.")


if __name__ == "__main__":
    asyncio.run(main())
