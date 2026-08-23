from symvion.config.models import TenantConfig, AgentRegistration
from symvion.agents.registry import AgentRegistry
from symvion.tools.registry import ToolRegistry
from symvion.core.graph import GraphFactory, GraphState
from symvion.memory.memory_store import get_memory_store
from symvion.core.context import TenantContext
from symvion.utils.helpers import ensure_messages, calculate_usage
from symvion.utils.security import (
    ALLOWED_PROVIDER_ENV_KEYS,
    apply_provider_env_vars,
    client_safe_error,
    sanitize_request_metadata,
)
from symvion.core.logger import logger
from symvion.core.exceptions import AgentExecutionError, RoutingError
from symvion.providers.factory import ProviderFactory
from symvion.core.middleware import MiddlewareChain, PIIMasker, UsageValidator, HallucinationValidator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TYPE_CHECKING, Dict, Any, Optional, Union, List, AsyncIterator
import time
import os

if TYPE_CHECKING:
    from symvion.rag.base import BaseRetriever

class Symvion:
    """
    Main entry point for the Symvion AI Orchestration Platform.
    Refactored in v0.3 for production-grade multi-tenancy and observability.
    """
    def __init__(self, config: TenantConfig, checkpointer: Optional[Any] = None):
        self.config = config
        # Injectable LangGraph checkpointer. Default (None) → process-local MemorySaver
        # at compile time. Multi-replica / Coronation prod should pass a durable saver.
        self.checkpointer = checkpointer
        
        # Only apply known provider credential keys when unset — never dump arbitrary
        # tenant env into process-global os.environ (cross-tenant bleed risk).
        applied = apply_provider_env_vars(self.config.env_vars)
        if applied:
            logger.info(
                "PROVIDER_ENV_APPLIED",
                None,
                tenant_id=self.config.tenant_id,
                keys=applied,
            )
        skipped = [
            k for k in (self.config.env_vars or {}) if k not in ALLOWED_PROVIDER_ENV_KEYS
        ]
        if skipped:
            logger.warning(
                "PROVIDER_ENV_SKIPPED_UNTRUSTED_KEYS",
                None,
                tenant_id=self.config.tenant_id,
                keys=skipped,
            )

        # Map generic API_KEY to provider-specific keys if set and target unset
        provider = (self.config.llm_provider or "openai").lower()
        api_key_val = os.environ.get("API_KEY")
        if api_key_val:
            if "anthropic" in provider and not os.environ.get("ANTHROPIC_API_KEY"):
                os.environ["ANTHROPIC_API_KEY"] = api_key_val
            elif "openai" in provider and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = api_key_val
            elif ("google" in provider or "gemini" in provider) and not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = api_key_val

        if self.config.interrupt_before_tools and checkpointer is None:
            logger.warning(
                "HITL_MEMORYSAVER_DEFAULT",
                None,
                tenant_id=self.config.tenant_id,
                message=(
                    "HITL is enabled but no durable checkpointer was provided; "
                    "MemorySaver is process-local and unsafe for multi-replica resume."
                ),
            )
                
        # Inject guardrails into base LLM config for propagation to all agents
        llm_config = self.config.llm_config.copy()
        llm_config["max_tokens_per_turn"] = self.config.max_tokens_per_turn

        self.agents = AgentRegistry(
            tenant_id=self.config.tenant_id, 
            llm_provider=self.config.llm_provider,
            llm_config=llm_config,
            streaming=self.config.streaming,
            enabled_agents=self.config.enabled_agents,
            agent_configs=self.config.agent_configs,
            allowed_outbound_hosts=self.config.allowed_outbound_hosts,
            file_sandbox_dir=self.config.file_sandbox_dir,
            backend_url=self.config.backend_url,
            auth_token=self.config.auth_token,
        )
        self.tools = ToolRegistry(iam_policies=self.config.iam_policies)
        self.memory_store = get_memory_store(self.config.tenant_id, self.config.memory)

        # Per-tenant retriever registry (populated via set_retriever)
        self._retrievers: Dict[str, "BaseRetriever"] = {}

        self.graph = None
        
        # Apply logging configuration
        logger.configure(
            enabled=self.config.logging.enabled,
            level=self.config.logging.level,
            file_path=self.config.logging.file_path,
            show_json=self.config.logging.show_json
        )

        # Initialize Middleware Chain (Stage 1)
        enabled_mw = self.config.enabled_middleware
        middleware = []
        
        if "pii" in enabled_mw:
            middleware.append(PIIMasker(self.config.governance.pii_patterns))
            
        if "usage" in enabled_mw:
            middleware.append(UsageValidator())
            
        if "policies" in enabled_mw:
            # Convert Pydantic models to dicts for the middleware
            policies_dict = [p.model_dump() for p in self.config.governance.policies]
            from symvion.core.middleware import PolicyMiddleware
            middleware.append(PolicyMiddleware(policies_dict))
            
        if self.config.enable_hallucination_check or "hallucination" in enabled_mw:
            middleware.append(HallucinationValidator(self.config.llm_provider, self.config.llm_config))
            
        self.middleware = MiddlewareChain(middleware)

        logger.info("ENGINE_INITIALIZED", None, tenant_id=self.config.tenant_id)
        
    def register_agent(self, payload: Union[Dict[str, Any], AgentRegistration, Any]):
        """
        Register a new agent. Supports registration dicts, AgentRegistration models, or direct Agent objects.
        """
        if isinstance(payload, dict):
            if "name" in payload and (hasattr(payload.get("agent_class"), "process") or "agent_class" in payload):
                # Custom object registration via dict
                name = payload["name"]
                agent_class = payload.get("agent_class")
                if agent_class:
                    config = self.agents.merge_agent_config(name, payload)
                    instance = agent_class(self.config.tenant_id, config)
                    self.agents._agents[name] = instance
                    logger.info("CUSTOM_AGENT_REGISTERED", None, tenant_id=self.config.tenant_id, agent_name=name)
                    self.graph = None
                    return
            registration = AgentRegistration(**payload)
        elif isinstance(payload, AgentRegistration):
            registration = payload
        else:
            # Direct object registration (assuming it has a 'name' attribute)
            name = getattr(payload, "name", getattr(payload, "__name__", "unknown"))
            self.agents._agents[name] = payload
            logger.info("DIRECT_AGENT_REGISTERED", None, tenant_id=self.config.tenant_id, agent_name=name)
            self.graph = None
            return
            
        self.agents.register_via_api(registration)
        logger.info("AGENT_REGISTERED", None, tenant_id=self.config.tenant_id, agent_name=registration.name)
        self.graph = None # Require recompile

    # ------------------------------------------------------------------
    # RAG integration
    # ------------------------------------------------------------------

    def set_retriever(self, tenant_id: str, retriever: "BaseRetriever") -> None:
        """
        Register a retriever for a specific tenant.

        Multiple tenants can have independent retrievers backed by
        completely different providers or data stores::

            runtime.set_retriever("insurance", vertex_retriever)
            runtime.set_retriever("hr", opensearch_retriever)

        Args:
            tenant_id:  The tenant to associate this retriever with.
            retriever:  Any :class:`~symvion.rag.base.BaseRetriever` instance.
        """
        self._retrievers[tenant_id] = retriever
        logger.info(
            "RETRIEVER_REGISTERED",
            None,
            tenant_id=tenant_id,
            retriever_type=type(retriever).__name__,
        )

    def get_retriever(self, tenant_id: str) -> "BaseRetriever":
        """
        Retrieve the retriever registered for a given tenant.

        Args:
            tenant_id: Tenant whose retriever to fetch.

        Raises:
            KeyError: If no retriever has been registered for this tenant.
        """
        if tenant_id not in self._retrievers:
            raise KeyError(
                f"No retriever registered for tenant '{tenant_id}'. "
                "Call runtime.set_retriever() first."
            )
        return self._retrievers[tenant_id]

    def register_tool(
        self,
        tenant: str,
        name: str,
        tool: Any,
    ) -> None:
        """
        Register a tool object (e.g. RetrievalTool) for use by agents.

        The tool is stored in the shared ToolRegistry and can be invoked
        by any agent via ``context.tools.execute(name, input_data)``::

            runtime.register_tool(
                tenant="insurance",
                name="retrieve_knowledge",
                tool=RetrievalTool(retriever),
            )

        Args:
            tenant: Tenant context (logged; currently tools are stored in a
                    shared registry keyed by name).
            name:   Unique tool name agents will use to call it.
            tool:   Tool object with an async ``execute`` method.
        """
        self.tools.register_tool(name, tool)
        logger.info(
            "TOOL_REGISTERED",
            None,
            tenant_id=tenant,
            tool_name=name,
            tool_type=type(tool).__name__,
        )

    def _ensure_graph(self):
        if not self.graph:
            effective_config = self.config.model_copy(deep=True)
            effective_config.llm_config["max_tokens_per_turn"] = self.config.max_tokens_per_turn
            self.graph = GraphFactory.build_graph(
                self.config.tenant_id,
                self.agents,
                self.tools,
                effective_config,
                checkpointer=self.checkpointer,
            )
        return self.graph

    def _sanitize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return sanitize_request_metadata(
            metadata,
            trust_client_role=bool(getattr(self.config, "trust_client_role", False)),
            trust_client_agent_id=bool(getattr(self.config, "trust_client_agent_id", False)),
            default_user_role=getattr(self.config, "default_user_role", "user") or "user",
        )

    def _thread_id(self, tenant: str, session_id: str) -> str:
        """Tenant-scoped LangGraph thread id to avoid cross-tenant session collisions."""
        return f"{tenant}:{session_id}"

    def _resolve_interrupt_allowlist(
        self, interrupt_before_tools: Optional[List[str]]
    ) -> List[str]:
        if interrupt_before_tools is not None:
            return list(interrupt_before_tools)
        return list(getattr(self.config, "interrupt_before_tools", None) or [])

    def _run_config(
        self,
        tenant: str,
        session_id: str,
        interrupt_before_tools: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        allowlist = self._resolve_interrupt_allowlist(interrupt_before_tools)
        meta: Dict[str, Any] = {}
        if extra_metadata:
            meta.update(extra_metadata)
        # Host-owned keys always win over request metadata.
        meta["tenant_id"] = tenant
        meta["session_id"] = session_id
        meta["interrupt_before_tools"] = allowlist
        meta["trust_hitl_edit"] = bool(getattr(self.config, "trust_hitl_edit", False))
        return {
            "configurable": {"thread_id": self._thread_id(tenant, session_id)},
            "metadata": meta,
            "recursion_limit": self.config.recursion_limit,
        }

    @staticmethod
    def _extract_interrupts(snapshot: Any) -> List[Any]:
        interrupts: List[Any] = []
        if snapshot is None:
            return interrupts
        raw = getattr(snapshot, "interrupts", None)
        if raw:
            interrupts.extend(list(raw))
        tasks = getattr(snapshot, "tasks", None) or ()
        for task in tasks:
            task_interrupts = getattr(task, "interrupts", None) or ()
            interrupts.extend(list(task_interrupts))
        return interrupts

    @staticmethod
    def _interrupt_payload(intr: Any) -> Dict[str, Any]:
        value = getattr(intr, "value", None)
        if value is None and isinstance(intr, dict):
            value = intr
        if not isinstance(value, dict):
            value = {"value": value}
        payload = {
            "type": "interrupt",
            "interrupt_id": getattr(intr, "id", None) or value.get("call_id"),
            "reason": value.get("reason", "tool_approval"),
            "tool": value.get("tool"),
            "call_id": value.get("call_id"),
            "args": value.get("args", {}),
            "ui": value.get("ui"),
        }
        return payload

    async def _stream_graph_events(
        self,
        graph_input: Any,
        run_config: Dict[str, Any],
        ctx: TenantContext,
        *,
        persist_history: Optional[List[Any]] = None,
        session_id: str = "default",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Shared streaming loop for chat_stream and resume_stream."""
        from symvion.tools.hitl import SYMVION_TOOL_START, SYMVION_TOOL_END

        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        final_result = None
        saw_tool_start_for: set = set()

        try:
            async for event in self.graph.astream_events(
                graph_input,
                version="v2",
                config=run_config,
            ):
                kind = event.get("event")
                name = event.get("name")
                data = event.get("data", {})

                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        content = ""
                        if hasattr(chunk, "content"):
                            content = chunk.content
                        elif isinstance(chunk, dict):
                            content = chunk.get("content", "")
                        if content:
                            yield {"type": "token", "content": content}

                elif kind == "on_chat_model_end":
                    output = data.get("output")
                    usage = None
                    if hasattr(output, "usage_metadata"):
                        usage = output.usage_metadata
                    elif isinstance(output, dict):
                        usage = output.get("usage_metadata")
                    if usage and isinstance(usage, dict):
                        for key in ["input_tokens", "output_tokens", "total_tokens"]:
                            total_usage[key] += usage.get(key, 0)

                elif kind == "on_chain_start" and name and name.endswith("_agent"):
                    yield {"type": "agent_start", "agent": name}

                elif kind == "on_tool_start":
                    call_key = f"{name}:{data.get('input')}"
                    saw_tool_start_for.add(call_key)
                    yield {
                        "type": "tool_start",
                        "tool": name,
                        "inputs": data.get("input"),
                    }

                elif kind == "on_tool_end":
                    yield {
                        "type": "tool_end",
                        "tool": name,
                        "output": data.get("output"),
                        "error": None,
                    }

                elif kind == "on_custom_event":
                    custom_name = name or data.get("name")
                    payload = data if isinstance(data, dict) else {}
                    # LangChain may nest custom data under "data"
                    if "data" in payload and isinstance(payload["data"], dict):
                        payload = payload["data"]
                    if custom_name == SYMVION_TOOL_START or event.get("name") == SYMVION_TOOL_START:
                        tool = payload.get("tool")
                        call_id = payload.get("call_id")
                        saw_tool_start_for.add(f"{tool}:{call_id}")
                        yield {
                            "type": "tool_start",
                            "tool": tool,
                            "inputs": payload.get("inputs"),
                            "call_id": call_id,
                        }
                    elif custom_name == SYMVION_TOOL_END or event.get("name") == SYMVION_TOOL_END:
                        yield {
                            "type": "tool_end",
                            "tool": payload.get("tool"),
                            "call_id": payload.get("call_id"),
                            "output": payload.get("output"),
                            "error": payload.get("error"),
                        }

                if kind == "on_chain_end" and not event.get("parent_ids"):
                    final_result = data.get("output")

            # Detect HITL pause
            snapshot = self.graph.get_state(run_config)
            interrupts = self._extract_interrupts(snapshot)
            if interrupts:
                for intr in interrupts:
                    payload = self._interrupt_payload(intr)
                    # Ensure tool_start preceded interrupt for inspectors
                    key = f"{payload.get('tool')}:{payload.get('call_id')}"
                    if key not in saw_tool_start_for and payload.get("tool"):
                        yield {
                            "type": "tool_start",
                            "tool": payload.get("tool"),
                            "inputs": payload.get("args"),
                            "call_id": payload.get("call_id"),
                        }
                    yield payload
                yield {"type": "done", "status": "interrupted"}
                return

            tokens_used = calculate_usage(total_usage)
            self.config.current_usage += tokens_used
            self.config.save()

            logger.info(
                "STREAM_USAGE_METRICS_UPDATED",
                ctx,
                tokens_used=tokens_used,
                total_usage=self.config.current_usage,
            )

            stream_meta = {
                "type": "metadata",
                "agent": final_result.get("current_agent", "default") if final_result else "default",
                "tokens": total_usage,
                "total_usage": self.config.current_usage,
                "request_id": ctx.request_id,
            }
            stream_meta = await self.middleware.run_postprocess(ctx, stream_meta)
            yield stream_meta

            if final_result and "agent_response" in final_result and final_result["agent_response"]:
                yield {"type": "done", "status": "success"}
                if persist_history is not None:
                    new_history = list(persist_history) + [
                        AIMessage(content=final_result.get("agent_response", ""))
                    ]
                    await self.memory_store.set(
                        session_id,
                        {
                            "history": new_history,
                            "current_agent": final_result.get("current_agent", "default"),
                            "summary": final_result.get("summary", ""),
                            "tools_used": final_result.get("tools_called", False),
                        },
                    )
            elif not final_result:
                # Resumed runs may still produce a values-shaped state
                values = getattr(snapshot, "values", None) if snapshot else None
                if values and values.get("agent_response"):
                    yield {
                        "type": "metadata",
                        "agent": values.get("current_agent", "default"),
                        "tokens": total_usage,
                        "total_usage": self.config.current_usage,
                        "request_id": ctx.request_id,
                    }
                    yield {"type": "done", "status": "success"}
                    if persist_history is not None:
                        new_history = list(persist_history) + [
                            AIMessage(content=values.get("agent_response", ""))
                        ]
                        await self.memory_store.set(
                            session_id,
                            {
                                "history": new_history,
                                "current_agent": values.get("current_agent", "default"),
                                "summary": values.get("summary", ""),
                                "tools_used": values.get("tools_called", False),
                            },
                        )
                else:
                    yield {"type": "error", "message": "Graph reached end without final result"}

        except Exception as e:
            logger.error("CHAT_STREAM_ERROR", ctx, error=str(e), exc_info=True)
            yield {"type": "error", "message": client_safe_error(e)}

    async def chat(
        self,
        tenant: str,
        message: str = "",
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        *,
        interrupt_before_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Main chat execution entry point with full observability.

        When a tool on the HITL allowlist pauses the graph, returns
        ``status: "interrupted"`` with an ``interrupt`` payload instead of
        completing the turn.
        """
        if tenant != self.config.tenant_id:
            raise ValueError(f"Tenant mismatch. Expected {self.config.tenant_id}, got {tenant}")
            
        # Initialize Request Context (strip untrusted client security fields by default)
        ctx_meta = self._sanitize_metadata(metadata)
             
        ctx = TenantContext(
            tenant_id=tenant, 
            metadata={
                **ctx_meta,
                "last_query": message,
                "iam_policies": self.config.iam_policies,
                "token_limit": self.config.token_limit,
                "current_usage": self.config.current_usage
            }
        )
        logger.info("REQUEST_START", ctx, session_id=session_id)
        
        # Run Middleware Preprocessing
        ctx, data = await self.middleware.run_preprocess(ctx, {"message": message})
        message = data["message"]
        
        self._ensure_graph()

        # Retrieve session history from memory_store
        session_data = await self.memory_store.get(session_id) or {"history": [], "current_agent": ctx_meta.get("agent_id", "default"), "summary": "", "tools_used": False}
        history = session_data.get("history", [])
        last_agent = ctx_meta.get("agent_id") or session_data.get("current_agent", "default")
        current_summary = session_data.get("summary", "")
        tools_used_in_session = session_data.get("tools_used", False)
        
        # Prepare messages
        current_message = HumanMessage(content=message)
        all_messages = history + [current_message]

        initial_state = GraphState(
            messages=all_messages,
            tenant_id=tenant,
            conversation_id=session_id,
            current_agent=last_agent,
            agent_response="",
            agent_type=last_agent,
            tool_calls=[],
            context={},
            cross_sell_opportunity={},
            previous_agent=last_agent,
            token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            summary=current_summary,
            tenant_context=ctx
        )

        run_config = self._run_config(tenant, session_id, interrupt_before_tools, ctx_meta)
        
        try:
            result = await self.graph.ainvoke(initial_state, config=run_config)
        except Exception as e:
            # Some LangGraph versions surface interrupts as exceptions
            snapshot = self.graph.get_state(run_config)
            interrupts = self._extract_interrupts(snapshot)
            if interrupts:
                payload = self._interrupt_payload(interrupts[0])
                return {
                    "status": "interrupted",
                    "interrupt": payload,
                    "agent": (getattr(snapshot, "values", {}) or {}).get("current_agent", last_agent),
                    "request_id": ctx.request_id,
                }
            logger.error("REQUEST_FAILED", ctx, error=str(e))
            raise AgentExecutionError(client_safe_error(e)) from e

        snapshot = self.graph.get_state(run_config)
        interrupts = self._extract_interrupts(snapshot)
        if interrupts:
            payload = self._interrupt_payload(interrupts[0])
            return {
                "status": "interrupted",
                "interrupt": payload,
                "agent": (getattr(snapshot, "values", {}) or {}).get("current_agent", last_agent),
                "request_id": ctx.request_id,
            }
        
        # Save updated history (including assistant response)
        summ_happened = False
        if "agent_response" in result and result["agent_response"]:
            new_history = all_messages + [AIMessage(content=result["agent_response"])]
            
            # Check for summarization threshold
            # If tools were used, threshold is 5 (per user request). Otherwise use global config.
            tools_called = result.get("tools_called", False)
            tools_used_in_session = tools_used_in_session or tools_called
            threshold = 5 if tools_used_in_session else self.config.summary_threshold
            
            if len(new_history) >= threshold:
                logger.info("SUMMARIZATION_START", ctx, threshold=threshold, tools_called=tools_called, tools_used_session=tools_used_in_session)
                current_summary = await self._summarize_history(ctx, new_history, current_summary)
                # Clear previous history and put summary as a new entry
                new_history = [SystemMessage(content=f"SUMMARY OF PREVIOUS CONVERSATION: {current_summary}")]
                summ_happened = True
                
            await self.memory_store.set(session_id, {
                "history": new_history,
                "current_agent": result.get("current_agent", "default"),
                "summary": current_summary,
                "tools_used": tools_used_in_session
            })
        
        # Thought Auditing (Reasoning Logging)
        if self.config.enable_thought_auditing and "agent_response" in result:
             logger.info("THOUGHT_AUDIT", ctx, 
                        agent=result.get("current_agent"), 
                        reasoning_length=len(result["agent_response"]))

        logger.info("REQUEST_SUCCESS", ctx, agent=result.get("current_agent"))
        
        # Finalize and Postprocess
        final_result = {
            "status": "success", 
            "data": result.get("agent_response", ""),
            "agent": result.get("current_agent", "unknown"),
            "tokens": result.get("token_usage", {}),
            "summarized": summ_happened,
            "request_id": ctx.request_id
        }
        
        final_result = await self.middleware.run_postprocess(ctx, final_result)
        
        return final_result

    async def chat_stream(
        self,
        tenant: str,
        message: str = "",
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        *,
        interrupt_before_tools: Optional[List[str]] = None,
    ):
        """
        Streamed chat execution entry point.

        Yields events: token, agent_start, tool_start, interrupt, tool_end,
        metadata, done (status success|interrupted), error.

        ``interrupt_before_tools`` is an allowlist of tool names that pause for
        GenUI/HITL. Empty/omitted uses TenantConfig default (empty = off).
        """
        if tenant != self.config.tenant_id:
            raise ValueError(f"Tenant mismatch. Expected {self.config.tenant_id}, got {tenant}")
            
        ctx_meta = self._sanitize_metadata(metadata)

        ctx = TenantContext(
            tenant_id=tenant, 
            metadata={
                **ctx_meta,
                "last_query": message,
                "iam_policies": self.config.iam_policies,
                "token_limit": self.config.token_limit,
                "current_usage": self.config.current_usage
            }
        )
        logger.info("STREAM_REQUEST_START", ctx, session_id=session_id)

        # Run Middleware Preprocessing
        ctx, data = await self.middleware.run_preprocess(ctx, {"message": message})
        message = data["message"]
        
        self._ensure_graph()

        session_data = await self.memory_store.get(session_id) or {"history": [], "current_agent": ctx_meta.get("agent_id", "default"), "summary": "", "tools_used": False}
        history = session_data.get("history", [])
        last_agent = ctx_meta.get("agent_id") or session_data.get("current_agent", "default")
        current_summary = session_data.get("summary", "")
        
        current_message = HumanMessage(content=message)
        all_messages = history + [current_message]

        initial_state = GraphState(
            messages=all_messages,
            tenant_id=tenant,
            conversation_id=session_id,
            current_agent=last_agent,
            agent_response="",
            agent_type=last_agent,
            tool_calls=[],
            context={},
            cross_sell_opportunity={},
            previous_agent=last_agent,
            token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            summary=current_summary,
            tenant_context=ctx
        )

        run_config = self._run_config(tenant, session_id, interrupt_before_tools, ctx_meta)
        async for evt in self._stream_graph_events(
            initial_state,
            run_config,
            ctx,
            persist_history=all_messages,
            session_id=session_id,
        ):
            yield evt

    async def resume_stream(
        self,
        tenant: str,
        session_id: str,
        resume: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Continue a graph paused on HITL interrupt.

        ``resume`` is typically::

            {"action": "approve"|"edit"|"reject"|"provide_result",
             "args": {...},   # for edit
             "result": ...}   # for provide_result / reject message

        Hosts must authenticate callers before invoking resume. ``provide_result``
        is only accepted for client/UI-marked tools.
        """
        if tenant != self.config.tenant_id:
            raise ValueError(f"Tenant mismatch. Expected {self.config.tenant_id}, got {tenant}")

        from langgraph.types import Command

        ctx_meta = self._sanitize_metadata(metadata)

        ctx = TenantContext(
            tenant_id=tenant,
            metadata={
                **ctx_meta,
                "iam_policies": self.config.iam_policies,
                "token_limit": self.config.token_limit,
                "current_usage": self.config.current_usage,
            },
        )
        logger.info("RESUME_STREAM_START", ctx, session_id=session_id)

        self._ensure_graph()
        # Preserve allowlist from the paused run when possible
        provisional = self._run_config(tenant, session_id, None, ctx_meta)
        snapshot = self.graph.get_state(provisional)
        prev_meta = {}
        if snapshot is not None:
            prev_cfg = getattr(snapshot, "config", None) or {}
            prev_meta = (prev_cfg.get("metadata") if isinstance(prev_cfg, dict) else {}) or {}
        allow = prev_meta.get("interrupt_before_tools")
        run_config = self._run_config(
            tenant,
            session_id,
            list(allow) if allow is not None else None,
            ctx_meta,
        )

        session_data = await self.memory_store.get(session_id) or {"history": []}
        history = session_data.get("history", [])

        async for evt in self._stream_graph_events(
            Command(resume=resume),
            run_config,
            ctx,
            persist_history=history,
            session_id=session_id,
        ):
            yield evt


    async def _summarize_history(self, context: TenantContext, history: list, existing_summary: str = "") -> str:
        """Use configured LLM to summarize conversation history."""
        llm = ProviderFactory.get_provider(self.config.llm_provider, {**self.config.llm_config, "temperature": 0})
        
        history_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in history])
        
        prompt = f"""Summarize the conversation history concisely. 
        Existing summary: {existing_summary}
        
        CONVERSATION:
        {history_str}
        
        NEW SUMMARY:"""
        
        res = await llm.ainvoke([SystemMessage(content=prompt)])
        return res.content
