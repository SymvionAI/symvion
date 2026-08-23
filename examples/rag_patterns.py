"""
examples/rag_patterns.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Two production-ready RAG patterns for Symvion:

  Pattern A — LLM-Aware Retrieval (inside an agent)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  The LLM decides when to call the retrieval tool based on the user's
  message. If the query needs knowledge-base context, the LLM emits a
  tool call; otherwise it answers directly from its own knowledge.
  This follows the exact same bind_tools + astream pattern used by the
  existing ClaimsAgent.

  Pattern B — API-Layer Retrieval (RAGPipeline)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  A FastAPI endpoint that unconditionally retrieves + generates for
  every request. Useful for dedicated search/Q&A endpoints where
  retrieval is always required.

Run from the ai-runtime directory:
    PYTHONPATH=src venv/bin/python3 examples/rag_patterns.py

Note: This demo uses a MockToolChatModel that simulates LLM tool-calling
so it runs without any API keys. In production, swap it for any real
provider (google_genai, openai, anthropic, etc.).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from symvion import (
    RAGPipeline,
    RetrievalTool,
    Symvion,
    TenantConfig,
    VertexRetriever,
)
from symvion.agents.base import BaseAgent
from symvion.core.context import TenantContext


# ===========================================================================
# MockToolChatModel — simulates LLM tool-calling without API keys
# ===========================================================================
# In production, replace this with a real provider:
#   ProviderFactory.get_provider("google_genai", {"model": "gemini-1.5-pro"})
#   ProviderFactory.get_provider("openai",       {"model": "gpt-4o"})
#
# The mock uses a simple heuristic: if the message looks like it needs
# knowledge-base data, it emits a tool_call; otherwise it answers directly.

_RETRIEVAL_KEYWORDS = {"policy", "cover", "coverage", "claim", "premium",
                       "procedure", "renew", "renewal", "flood", "document"}


class MockToolChatModel(BaseChatModel):
    """
    Minimal mock that supports bind_tools() and emits tool calls based
    on keyword detection. Only used for demo/testing — no real LLM involved.
    """

    bound_tools: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "mock-tool-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "MockToolChatModel":
        """Return a copy of this model with the tools registered."""
        clone = MockToolChatModel()
        clone.bound_tools = list(tools)
        return clone

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_human = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            "",
        )
        # If tools are bound and message contains retrieval keywords → tool call
        if self.bound_tools and any(kw in last_human.lower() for kw in _RETRIEVAL_KEYWORDS):
            tool_name = self.bound_tools[0].name if self.bound_tools else "retrieve_knowledge"
            ai_msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": {"query": last_human},
                    "id": "mock_call_001",
                    "type": "tool_call",
                }],
            )
        else:
            # Find last ToolMessage (retrieval result) and answer from it
            tool_result = next(
                (m.content for m in reversed(messages) if isinstance(m, ToolMessage)),
                None,
            )
            if tool_result:
                ai_msg = AIMessage(
                    content=f"Based on the retrieved information: {tool_result[:120]}..."
                )
            else:
                ai_msg = AIMessage(content="Mock answer based on general knowledge.")

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, **kwargs)

    async def astream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Yield the full response as a single chunk (simulates streaming)."""
        messages = input if isinstance(input, list) else [input]
        result = self._generate(messages)
        msg = result.generations[0].message
        chunk = AIMessageChunk(
            content=msg.content,
            tool_calls=getattr(msg, "tool_calls", []),
            tool_call_chunks=[
                {
                    "name": tc["name"],
                    "args": json.dumps(tc["args"]),
                    "id": tc["id"],
                    "index": i,
                    "type": "tool_call_chunk",
                }
                for i, tc in enumerate(getattr(msg, "tool_calls", []))
            ],
        )
        yield chunk



# ===========================================================================
# Shared runtime setup
# ===========================================================================

config = TenantConfig(
    tenant_id="acme",
    llm_provider="mock",
    llm_config={"model": "mock"},
    logging={"enabled": True, "level": "INFO", "show_json": False},
)

runtime = Symvion(config)

retriever = VertexRetriever(
    project_id="my-gcp-project",
    location="global",
    data_store_id=None,      # stub mode — swap for a real data store ID
    default_top_k=5,
)

retrieval_tool_obj = RetrievalTool(retriever, default_top_k=5)

runtime.set_retriever("acme", retriever)
runtime.register_tool(tenant="acme", name="retrieve_knowledge", tool=retrieval_tool_obj)


# ===========================================================================
# PATTERN A — LLM-Aware Retrieval (inside a BaseAgent)
# ===========================================================================
#
# Lifecycle per request:
#
#   User message
#       ↓
#   llm.bind_tools([retrieve_knowledge]).astream(messages)
#       ├── LLM emits tool_call  → execute retrieval → feed ToolMessage back
#       │                          → llm.astream(messages + ToolMessage) → final answer
#       └── LLM answers directly → return response immediately (no retrieval)
#
# In production, swap MockToolChatModel for a real provider and the LLM
# will decide based on semantic understanding, not keyword matching.

class KnowledgeAgent(BaseAgent):
    """
    General-purpose agent with LLM-aware retrieval.

    The LLM autonomously decides when to call 'retrieve_knowledge' based
    on whether the query requires information from the knowledge base.
    Follows the same bind_tools + astream pattern as ClaimsAgent.
    """

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        super().__init__(tenant_id, config)
        self.description = (
            "Answers questions using the knowledge base when needed. "
            "Use for policy queries, product information, and FAQs."
        )

        # Expose the RetrievalTool as a LangChain StructuredTool so the LLM
        # can call it natively via tool-calling (same as ClaimsAgent's tools).
        self._lc_retrieval_tool = StructuredTool.from_function(
            coroutine=self._retrieve,
            name="retrieve_knowledge",
            description=(
                "Search the knowledge base for relevant information. "
                "Call this when the user asks about policies, procedures, "
                "products, or any topic requiring factual documentation."
            ),
        )

    async def _retrieve(self, query: str) -> str:
        """Bridge: calls RetrievalTool and serialises docs for the ToolMessage."""
        result = await retrieval_tool_obj.execute({"query": query, "top_k": 5})
        docs = result["documents"]
        return json.dumps([
            {"text": d["text"], "source": d["metadata"]} for d in docs
        ])

    async def execute(
        self,
        context: TenantContext,
        input_data: Dict[str, Any],
        tools: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        message: str = input_data.get("message", "")
        history: list = input_data.get("history", [])

        # ── Build message list ─────────────────────────────────────────
        messages: List[BaseMessage] = [
            SystemMessage(content=(
                "You are a helpful assistant. "
                "Use the retrieve_knowledge tool when the user asks something "
                "that requires factual documentation from the knowledge base. "
                "Answer general questions directly without retrieval."
            )),
        ]

        for msg in history[-10:]:
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=message))

        # ── First LLM call — may or may not trigger tool use ──────────
        llm_with_tool = self.llm.bind_tools([self._lc_retrieval_tool])

        response = None
        async for chunk in llm_with_tool.astream(messages, config=config):
            response = chunk if response is None else response + chunk

        # ── Tool-call branch: LLM chose to retrieve ────────────────────
        if hasattr(response, "tool_calls") and response.tool_calls:
            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", "call_0")

                if tool_name == "retrieve_knowledge":
                    try:
                        raw = await self._retrieve(**tool_args)
                        messages.append(
                            ToolMessage(content=raw, tool_call_id=tool_call_id)
                        )
                    except Exception as exc:
                        messages.append(
                            ToolMessage(
                                content=json.dumps({"error": str(exc)}),
                                tool_call_id=tool_call_id,
                            )
                        )

            # ── Second LLM call: generate final answer with context ────
            final_response = None
            async for chunk in self.llm.astream(messages, config=config):
                final_response = chunk if final_response is None else final_response + chunk

            return {
                "agent_response": final_response.content,
                "retrieval_used": True,
                "token_usage": getattr(final_response, "usage_metadata", {}),
            }

        # ── No tool call: LLM answered directly ───────────────────────
        return {
            "agent_response": response.content,
            "retrieval_used": False,
            "token_usage": getattr(response, "usage_metadata", {}),
        }


# Register with the runtime using the mock LLM
_knowledge_agent = KnowledgeAgent(
    tenant_id="acme",
    config={"name": "knowledge", "provider": "mock", "model": "mock"},
)
# Inject our tool-capable mock LLM into the agent
_knowledge_agent.llm = MockToolChatModel()
runtime.register_agent(_knowledge_agent)


# ===========================================================================
# PATTERN B — API-Layer Retrieval (RAGPipeline)
# ===========================================================================
#
# Retrieval + generation happen unconditionally on every request.
# The caller hits /search and always gets KB-grounded answers.
# No LLM routing or agent involved — just retriever → prompt → LLM.
#
# In production: sits in your FastAPI app, built once at startup.

_pipeline_llm = MockToolChatModel()   # swap: ProviderFactory.get_provider(...)

pipeline = RAGPipeline(
    retriever=retriever,
    llm=_pipeline_llm,
    top_k=5,
    fallback_response="I could not find relevant information to answer that.",
    # Optional: inject a domain-specific prompt builder from outside Symvion:
    # prompt_builder=lambda query, docs: f"<your template>\n{query}\n{docs}"
)


async def search_endpoint(query: str) -> Dict[str, Any]:
    """
    Simulates a FastAPI search handler.

    In production:
        @app.post("/search")
        async def search(query: str):
            return await search_endpoint(query)
    """
    output = await pipeline.run(query)
    return {
        "answer": output["answer"],
        "sources": [d["metadata"] for d in output["documents"]],
        "query": output["query"],
        "duration_s": output["duration_s"],
    }


# ===========================================================================
# Demo runner
# ===========================================================================

async def main() -> None:

    ctx = TenantContext(tenant_id="acme", metadata={"user_role": "user"})

    # ── Pattern A ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("PATTERN A — LLM-Aware Retrieval (KnowledgeAgent)")
    print("=" * 62)
    print("The LLM decides per-query whether retrieval is needed.")

    # Query 1: contains policy keywords → LLM will emit a tool call
    q1 = "What does our flood damage policy cover?"
    print(f"\n[Q1] {q1!r}")
    r1 = await _knowledge_agent.run(ctx, {"message": q1})
    print(f"  retrieval_used : {r1['retrieval_used']}")
    print(f"  answer         : {r1['agent_response']}")

    # Query 2: general knowledge → LLM answers directly (no retrieval)
    q2 = "What is the capital of France?"
    print(f"\n[Q2] {q2!r}")
    r2 = await _knowledge_agent.run(ctx, {"message": q2})
    print(f"  retrieval_used : {r2['retrieval_used']}")
    print(f"  answer         : {r2['agent_response']}")

    # ── Pattern B ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("PATTERN B — API-Layer Retrieval (RAGPipeline / search endpoint)")
    print("=" * 62)
    print("Every request retrieves unconditionally.\n")

    api_queries = [
        "How do I renew my policy?",
        "What are the premium payment options?",
    ]

    for q in api_queries:
        print(f"[Q] {q!r}")
        resp = await search_endpoint(q)
        print(f"  answer   : {resp['answer']}")
        print(f"  sources  : {resp['sources']}")
        print(f"  duration : {resp['duration_s']}s\n")

    print("✅  All pattern demos complete.")


if __name__ == "__main__":
    asyncio.run(main())
