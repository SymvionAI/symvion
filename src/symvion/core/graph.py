import os
import logging
import time
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from symvion.agents.registry import AgentRegistry
from symvion.tools.registry import ToolRegistry
from symvion.config.models import AgentRegistration, TenantConfig
from symvion.core.context import TenantContext
from symvion.core.logger import logger
from symvion.providers.factory import ProviderFactory
from symvion.tools.base import ToolSafetyWrapper
from symvion.utils.helpers import ensure_messages, merge_usage
from symvion.routers.default import KeywordRouter, LLMRouter

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], "Conversation messages"]
    tenant_id: str
    conversation_id: str
    current_agent: str
    agent_response: str
    agent_type: str
    tool_calls: List[Dict[str, Any]]
    context: Dict[str, Any]
    cross_sell_opportunity: Dict[str, Any]
    previous_agent: str
    token_usage: NotRequired[Dict[str, Any]]
    summary: NotRequired[str]
    tenant_context: TenantContext # Structured context
    tools_called: NotRequired[bool]

class GraphFactory:
    @staticmethod
    def build_graph(
        tenant_id: str,
        agents: AgentRegistry,
        tools: ToolRegistry,
        tenant_config: TenantConfig,
        checkpointer: Optional[Any] = None,
    ) -> StateGraph:
        # Resolve LLM using ProviderFactory
        llm_config = tenant_config.llm_config.copy()
        if "streaming" not in llm_config:
            llm_config["streaming"] = tenant_config.streaming
        if "max_tokens_per_turn" not in llm_config:
            llm_config["max_tokens_per_turn"] = tenant_config.max_tokens_per_turn
            
        llm = ProviderFactory.get_provider(tenant_config.llm_provider, llm_config)
        
        workflow = StateGraph(GraphState)
        
        # Router node
        workflow.add_node("router", GraphFactory._create_router_node(llm, agents, tenant_config.router_type))
        
        # Load all agents
        registered_agent_keys = agents.get_all_agent_names()
        for agent_name in registered_agent_keys:
            agent_obj = agents.get_agent(agent_name)
            workflow.add_node(
                f"{agent_name}_agent",
                GraphFactory._create_agent_node(agent_name, agent_obj, llm, tools)
            )

        workflow.add_node("default_agent", GraphFactory._create_default_agent_node(llm, tenant_id, tenant_config, agents))
        workflow.add_node("tool_executor", GraphFactory._create_tool_executor_node(tools))
        workflow.add_node("final_response", GraphFactory._create_final_response_node())

        workflow.set_entry_point("router")

        # Routing edges
        routing_map = {name: f"{name}_agent" for name in registered_agent_keys}
        routing_map["general"] = "default_agent"
        routing_map["tools"] = "tool_executor"
        workflow.add_conditional_edges("router", lambda x: x.get("current_agent", "general"), routing_map)

        # Agent edges
        for name in registered_agent_keys:
            workflow.add_edge(f"{name}_agent", "final_response")
        workflow.add_edge("default_agent", "final_response")
        workflow.add_edge("tool_executor", "final_response")

        workflow.add_edge("final_response", END)

        # Default MemorySaver is process-local (fine for tests/single-process).
        # Inject a durable checkpointer for multi-replica / Coronation prod.
        if checkpointer is None:
            try:
                from langgraph.checkpoint.memory import MemorySaver
            except ImportError:  # pragma: no cover
                from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
            checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    @staticmethod
    def _create_router_node(llm, agents: AgentRegistry, router_type: str):
        # Initialize the router based on type
        if router_type == "llm":
            router = LLMRouter(llm)
        else:
            router = KeywordRouter()
            
        async def router_node(state: GraphState, config: RunnableConfig) -> GraphState:
            ctx = state["tenant_context"]
            
            # Check for direct agent target in metadata (override routing)
            target_agent = ctx.metadata.get("agent_id")
            if target_agent and target_agent not in ("general", "default"):
                agent_names = agents.get_all_agent_names()
                if target_agent in agent_names:
                    logger.info("ROUTING_BYPASS", ctx, agent_selected=target_agent, strategy="metadata_override")
                    return {
                        **state,
                        "current_agent": target_agent,
                        "agent_type": target_agent,
                        "token_usage": {}
                    }

            messages = state["messages"]
            last_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
            
            agent_names = agents.get_all_agent_names()
            
            if router_type == "llm":
                # For LLM routing, we need descriptions
                agent_info = []
                for name in agent_names:
                    agent_obj = agents.get_agent(name)
                    desc = "No description available."
                    if hasattr(agent_obj, "description"):
                        desc = agent_obj.description
                    elif isinstance(agent_obj, dict):
                        desc = agent_obj.get("description", desc)
                    agent_info.append({
                        "name": name,
                        "description": desc
                    })
                
                # pyrefly: ignore [unexpected-keyword]
                route_result = await router.route(ctx, last_message, agent_info, config=config)
            else:
                # Standard keyword routing
                route_result = await router.route(ctx, last_message, agent_names)
            
            selected_agent = route_result.get("agent", "general")
            usage = route_result.get("token_usage", {})
            
            return {
                **state, 
                "current_agent": selected_agent, 
                "agent_type": selected_agent,
                "token_usage": usage
            }
        return router_node

    @staticmethod
    def _create_agent_node(agent_name: str, agent_obj: Any, llm: Any, tools: ToolRegistry):
        async def agent_node(state: GraphState, config: RunnableConfig) -> GraphState:
            ctx = state["tenant_context"]
            messages = state["messages"]
            
            from symvion.agents.base import BaseAgent
            
            input_data = {
                "message": messages[-1].content if messages else "",
                "history": messages[:-1],
                "context": state.get("context", {})
            }
            
            tools_called = False
            if isinstance(agent_obj, BaseAgent):
                # New v0.3 Agent with Lifecycle and Registry-based Tools
                result = await agent_obj.run(ctx, input_data, tools=tools, config=config)
                response_text = result.get("agent_response", "")
                usage = result.get("token_usage") or result.get("usage_metadata") or result.get("usage") or {}
                tools_called = result.get("tools_called", False)
            else:
                # Legacy / Dynamic Agent support
                logger.info("LEGACY_AGENT_INVOKED", ctx, agent_name=agent_name)
                if hasattr(agent_obj, "process"):
                    # Process takes dict and returns dict
                    res = await agent_obj.process(input_data, config=config)
                    response_text = res.get("agent_response", res.get("content", res.get("response", "")))
                    usage = res.get("token_usage") or res.get("usage_metadata") or res.get("usage") or {}
                    tools_called = res.get("tools_called", False)
                else:
                    # Pure dynamic registration (AgentRegistration model or dict)
                    if isinstance(agent_obj, dict):
                        system_prompt = agent_obj.get("system_prompt", "You are a helpful assistant.")
                        agent_tools_names = agent_obj.get("tools", [])
                    else:
                        system_prompt = getattr(agent_obj, "system_prompt", "You are a helpful assistant.")
                        agent_tools_names = getattr(agent_obj, "tools", [])
                    
                    sys_msg = SystemMessage(content=system_prompt)
                    llm_messages = [sys_msg] + ensure_messages(messages)
                    
                    # Load actual langchain tools
                    from symvion.utils.helpers import filter_allowed_tools
                    from symvion.tools.base import ToolSafetyWrapper

                    loaded_tools = []
                    if agent_tools_names:
                        import importlib, pkgutil
                        # pyrefly: ignore [missing-import]
                        import tools as tools_pkg
                        for _, modname, _ in pkgutil.iter_modules(tools_pkg.__path__):
                            try:
                                mod = importlib.import_module(f"tools.{modname}")
                                for t_name in agent_tools_names:
                                    if hasattr(mod, t_name):
                                        t_func = getattr(mod, t_name)
                                        if hasattr(t_func, "name"):
                                            loaded_tools.append(t_func)
                            except Exception as e:
                                logger.error("TOOL_LOAD_ERROR", ctx, error=str(e))

                    # Enforce IAM before the model can see or call tools.
                    loaded_tools = filter_allowed_tools(ctx, loaded_tools)
                                
                    llm_with_tools = llm.bind_tools(loaded_tools) if loaded_tools else llm
                    
                    # Use astream for dynamic agents
                    response_text = ""
                    usage = {}
                    
                    response = None
                    async for chunk in llm_with_tools.astream(llm_messages, config=config):
                        if response is None: response = chunk
                        else: response += chunk
                        
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        tools_called = True
                        llm_messages.append(response)
                        
                        import json
                        import inspect
                        from symvion.tools.hitl import run_tool_with_hitl
                        for tool_call in response.tool_calls:
                            t_name = tool_call.get("name")
                            t_args = tool_call.get("args", {})
                            call_id = tool_call.get("id") or f"call_{t_name}"
                            
                            matched_tool = None
                            t_func = None
                            for t in loaded_tools:
                                if t.name == t_name:
                                    matched_tool = t
                                    t_func = getattr(t, "func", t)
                                    if hasattr(t_func, "func"):
                                        t_func = t_func.func
                                    break
                            
                            res_content = "Tool not found"
                            if t_func:
                                try:
                                    _func = t_func
                                    iam = ctx.metadata.get("iam_policies")

                                    async def _exec(_f=_func, _n=t_name, **kwargs):
                                        return await ToolSafetyWrapper.invoke(
                                            _f,
                                            ctx,
                                            _n,
                                            kwargs or {},
                                            iam_policies=iam,
                                        )

                                    res = await run_tool_with_hitl(
                                        tool_name=t_name,
                                        tool_args=t_args or {},
                                        call_id=call_id,
                                        execute=_exec,
                                        config=config,
                                        tool=matched_tool,
                                    )
                                    res_content = json.dumps(res) if not isinstance(res, str) else res
                                except Exception as e:
                                    # GraphInterrupt must propagate so chat_stream can
                                    # emit a GenUI interrupt event (do not swallow HITL).
                                    try:
                                        from langgraph.errors import GraphInterrupt
                                    except ImportError:  # pragma: no cover
                                        GraphInterrupt = ()  # type: ignore
                                    if GraphInterrupt and isinstance(e, GraphInterrupt):
                                        raise
                                    if type(e).__name__ in ("GraphInterrupt", "GraphBubbleUp"):
                                        raise
                                    res_content = "Error: tool execution failed"
                                    
                            from langchain_core.messages import ToolMessage
                            llm_messages.append(ToolMessage(content=res_content, tool_call_id=tool_call.get("id")))
                            
                        # Get final response
                        final_response = None
                        async for chunk in llm.astream(llm_messages, config=config):
                            if final_response is None: final_response = chunk
                            else: final_response += chunk
                            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                                usage = chunk.usage_metadata
                        response_text = final_response.content
                    else:
                        tools_called = False
                        response_text = response.content
                        if hasattr(response, "usage_metadata") and response.usage_metadata:
                            usage = response.usage_metadata

            # Aggregate usage with previous nodes (e.g. router) safely
            current_usage = state.get("token_usage", {})
            merged_usage = merge_usage(current_usage, usage)

            return {
                **state,
                "agent_response": response_text,
                "current_agent": agent_name,
                "agent_type": agent_name,
                "token_usage": merged_usage,
                "tools_called": tools_called,
                "previous_agent": state.get("current_agent", "")
            }
        return agent_node

    @staticmethod
    def _create_default_agent_node(llm, tenant_id, tenant_config, agents: AgentRegistry):
        async def default_agent_node(state: GraphState, config: RunnableConfig) -> GraphState:
            ctx = state["tenant_context"]
            messages = state["messages"]
            
            if tenant_config.system_prompt:
                system_prompt = tenant_config.system_prompt
            else:
                system_prompt = f"You are a general assistant for {tenant_id}. Help with greetings and basic questions."
                
            llm_messages = [SystemMessage(content=system_prompt)] + ensure_messages(messages)
            
            logger.info("DEFAULT_AGENT_INVOKED", ctx)
            
            # Use astream for default agent
            response_text = ""
            usage = {}
            async for chunk in llm.astream(llm_messages, config=config):
                response_text += chunk.content
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = chunk.usage_metadata
            
            # Aggregate usage safely
            current_usage = state.get("token_usage", {})
            merged_usage = merge_usage(current_usage, usage)

            return {
                **state,
                "agent_response": response_text,
                "current_agent": "default",
                "agent_type": "default",
                "token_usage": merged_usage
            }
        return default_agent_node

    @staticmethod
    def _create_tool_executor_node(registry):
        async def tool_node(state: GraphState) -> GraphState:
            ctx = state["tenant_context"]
            # Tool execution via Safe Wrapper (Simulated here for breadcrumb)
            logger.info("TOOL_EXECUTOR_INVOKED", ctx)
            return {**state, "agent_response": "Tools execution safety layer active."}
        return tool_node

    @staticmethod
    def _create_final_response_node():
        async def final_node(state: GraphState) -> GraphState:
            return state
        return final_node
