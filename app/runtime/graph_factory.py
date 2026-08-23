"""
Graph factory for building tenant-specific LangGraph instances.
Dynamically constructs graphs based on tenant configuration.
"""

import os
import time
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import logging

from app.runtime.agents.registration_agent import RegistrationAgent
from app.runtime.agents.claims_agent import ClaimsAgent
from app.runtime.agents.billing_agent import BillingAgent
from app.runtime.agents.complaint_agent import ComplaintAgent
from app.runtime.agents.insurance_agent import InsuranceAgent
from app.runtime.agents.quote_generator_agent import QuoteGeneratorAgent
from app.runtime.agents.document_intelligence_agent import DocumentIntelligenceAgent
from app.runtime.agents.legacy_agent import (
    doctrine_router_route,
    doctrine_engine_evaluate,
    LegacyAgent,
    empty_evaluation,
    get_principles,
)
from app.runtime.tools.tool_wrappers import ToolWrapper
from app.utils.logging import emit_doctrine_event

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State for the conversation graph."""

    messages: Annotated[List[BaseMessage], "Conversation messages"]
    tenant_id: str
    conversation_id: str
    current_agent: str
    agent_response: str
    agent_type: str  # Agent type for debugging/display (e.g., "claims", "billing")
    tool_calls: List[Dict[str, Any]]
    context: Dict[str, Any]
    cross_sell_opportunity: Dict[str, Any]  # Cross-sell detection result
    previous_agent: str  # Track previous agent for cross-sell handoff
    # Doctrine evaluation layer (set by doctrine_evaluation node)
    doctrine_evaluation: NotRequired[Dict[str, Any]]
    doctrine_route: NotRequired[str]


class GraphFactory:
    """Factory for creating tenant-specific LangGraph instances."""

    @staticmethod
    def build_graph(
        tenant_id: str,
        allowed_agents: List[str],
        allowed_tools: List[str],
        tenant_config: Dict[str, Any],
    ) -> StateGraph:
        """
        Build a LangGraph instance for a specific tenant.

        Args:
            tenant_id: Unique tenant identifier
            allowed_agents: List of agent types allowed for this tenant
            allowed_tools: List of tool identifiers allowed for this tenant
            tenant_config: Additional tenant-specific configuration

        Returns:
            Configured LangGraph instance
        """
        # Initialize LLM for routing and responses
        model_name = os.getenv("LLM_MODEL", "gpt-4")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        llm = ChatOpenAI(model=model_name, temperature=temperature)

        # Initialize agents (only those allowed for this tenant)
        agents = {}
        agent_config = tenant_config.get("agent_config", {})

        if not allowed_agents or "registration" in allowed_agents:
            agents["registration"] = RegistrationAgent(
                tenant_id, agent_config.get("registration", {})
            )

        if not allowed_agents or "claims" in allowed_agents:
            agents["claims"] = ClaimsAgent(tenant_id, agent_config.get("claims", {}))

        if not allowed_agents or "billing" in allowed_agents:
            agents["billing"] = BillingAgent(tenant_id, agent_config.get("billing", {}))

        if not allowed_agents or "complaint" in allowed_agents:
            agents["complaint"] = ComplaintAgent(
                tenant_id, agent_config.get("complaint", {})
            )

        if not allowed_agents or "insurance" in allowed_agents:
            agents["insurance"] = InsuranceAgent(
                tenant_id, agent_config.get("insurance", {})
            )

        if not allowed_agents or "quote_generator" in allowed_agents:
            agents["quote_generator"] = QuoteGeneratorAgent(
                tenant_id, agent_config.get("quote_generator", {})
            )

        # Initialize document_intelligence agent if it's in allowed_agents
        # Document processing now uses local OCR service, no MCP tool needed
        if not allowed_agents or "document_intelligence" in allowed_agents:
            agents["document_intelligence"] = DocumentIntelligenceAgent(
                tenant_id, agent_config.get("document_intelligence", {})
            )

        # Initialize tool wrapper
        tool_wrapper = ToolWrapper(tenant_id, allowed_tools or [])

        # Doctrine layer: LegacyAgent for strategic path
        legacy_agent = LegacyAgent(tenant_id, tenant_config or {})

        # Build the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("router", GraphFactory._create_router_node(llm, agents))
        workflow.add_node(
            "registration_agent",
            GraphFactory._create_agent_node("registration", agents.get("registration")),
        )
        workflow.add_node(
            "claims_agent",
            GraphFactory._create_agent_node("claims", agents.get("claims")),
        )
        workflow.add_node(
            "billing_agent",
            GraphFactory._create_agent_node("billing", agents.get("billing")),
        )
        workflow.add_node(
            "complaint_agent",
            GraphFactory._create_agent_node("complaint", agents.get("complaint")),
        )
        workflow.add_node(
            "insurance_agent",
            GraphFactory._create_agent_node("insurance", agents.get("insurance")),
        )
        workflow.add_node(
            "quote_generator_agent",
            GraphFactory._create_agent_node(
                "quote_generator", agents.get("quote_generator")
            ),
        )
        workflow.add_node(
            "document_intelligence_agent",
            GraphFactory._create_agent_node(
                "document_intelligence",
                agents.get("document_intelligence"),
                allowed_agents=allowed_agents or [],
            ),
        )
        workflow.add_node(
            "default_agent",
            GraphFactory._create_default_agent_node(llm, tenant_id, tenant_config),
        )
        workflow.add_node(
            "tool_executor", GraphFactory._create_tool_executor_node(tool_wrapper)
        )
        workflow.add_node(
            "cross_sell_detector",
            GraphFactory._create_cross_sell_detector_node(llm, agents),
        )
        workflow.add_node(
            "doctrine_evaluation",
            GraphFactory._create_doctrine_evaluation_node(legacy_agent),
        )
        workflow.add_node("final_response", GraphFactory._create_final_response_node())

        # Set entry point to default agent
        # The default agent handles conversations unless a specialized agent is explicitly needed
        workflow.set_entry_point("default_agent")

        # Add edges from router
        workflow.add_conditional_edges(
            "router",
            GraphFactory._route_to_agent,
            {
                "registration": "registration_agent",
                "claims": "claims_agent",
                "billing": "billing_agent",
                "complaint": "complaint_agent",
                "insurance": "insurance_agent",
                "quote_generator": "quote_generator_agent",
                "document_intelligence": "document_intelligence_agent",
                "general": "default_agent",
                "tools": "tool_executor",
            },
        )

        # All agent nodes go to cross-sell detector for opportunity detection
        workflow.add_edge("registration_agent", "cross_sell_detector")
        workflow.add_edge("claims_agent", "cross_sell_detector")
        workflow.add_edge("billing_agent", "cross_sell_detector")
        workflow.add_edge("complaint_agent", "cross_sell_detector")
        workflow.add_edge("insurance_agent", "cross_sell_detector")
        workflow.add_edge("quote_generator_agent", "cross_sell_detector")
        workflow.add_edge("document_intelligence_agent", "cross_sell_detector")
        workflow.add_edge("default_agent", "cross_sell_detector")
        workflow.add_edge("tool_executor", "cross_sell_detector")

        # Cross-sell detector routes to either another agent or doctrine (then final response)
        workflow.add_conditional_edges(
            "cross_sell_detector",
            GraphFactory._route_from_cross_sell,
            {
                "registration": "registration_agent",
                "claims": "claims_agent",
                "billing": "billing_agent",
                "complaint": "complaint_agent",
                "insurance": "insurance_agent",
                "quote_generator": "quote_generator_agent",
                "document_intelligence": "document_intelligence_agent",
                "general": "default_agent",
                "final": "doctrine_evaluation",
            },
        )

        # Doctrine evaluation runs before final response
        workflow.add_edge("doctrine_evaluation", "final_response")

        # Final response ends
        workflow.add_edge("final_response", END)

        # Compile the graph
        graph = workflow.compile()

        logger.info(
            f"Built LangGraph for tenant {tenant_id} with agents: {list(agents.keys())}"
        )

        return graph

    @staticmethod
    def create_initial_state(
        tenant_id: str,
        conversation_id: str,
        user_message: str,
        context: Dict[str, Any] = None,
        message_history: List[Dict[str, Any]] = None,
    ) -> GraphState:
        """
        Create initial state for graph execution.

        Args:
            tenant_id: Unique tenant identifier
            conversation_id: Unique conversation identifier
            user_message: User's message content
            context: Additional context
            message_history: Previous conversation messages

        Returns:
            Initial graph state
        """
        messages = []

        # Add message history if provided
        if message_history:
            for msg in message_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))

        # Add current user message
        messages.append(HumanMessage(content=user_message))

        return GraphState(
            messages=messages,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            current_agent="default",  # Default to default agent
            agent_response="",
            agent_type="default",  # Default agent type
            tool_calls=[],
            context=context or {},
            cross_sell_opportunity={},
            previous_agent="",
        )

    @staticmethod
    def _create_router_node(llm: ChatOpenAI, agents: Dict[str, Any]):
        """Create the router node that classifies user intent."""

        async def router_node(state: GraphState) -> GraphState:
            """Route message to appropriate agent based on intent."""
            messages = state["messages"]
            last_message = messages[-1] if messages else None

            if not last_message or not isinstance(last_message, HumanMessage):
                return {**state, "current_agent": "default", "agent_type": "default"}

            user_message = last_message.content

            # Use LLM to classify intent - be conservative, only route if there's a CLEAR need
            # The default agent (general) handles most conversations
            routing_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""You are a routing assistant. The default agent handles most conversations.
Only route to a specialized agent if the user's message has a VERY CLEAR and EXPLICIT intent for that specific domain.

Categories (only route if intent is EXPLICIT):
- insurance: User explicitly asks about insurance policies, coverage, or insurance advice
- quote_generator: User explicitly asks for a quote, pricing, or premium estimate
- registration: User explicitly wants to register, create an account, or onboard
- claims: User mentions ANY of: filing a claim, checking claim status, "my claim", "claim number", claim details, submitting a claim, or any claim-related query
- billing: User explicitly asks about billing, payments, invoices, or account balance
- complaint: User explicitly complains, reports an issue, or needs escalation
- document_intelligence: User explicitly asks about documents, wants to extract text from PDFs/images, analyze documents, or perform OCR
- general: Everything else - default behavior for most conversations

Available agents: {', '.join(agents.keys()) if agents else 'None'}

IMPORTANT: Default to "general" unless there's a VERY CLEAR and EXPLICIT intent for a specialized agent.
Most conversations should stay with the default agent.

Respond with ONLY the category name (insurance, quote_generator, registration, claims, billing, complaint, or general).""",
                    ),
                    ("human", "{message}"),
                ]
            )

            try:
                chain = routing_prompt | llm
                response = await chain.ainvoke({"message": user_message})
                agent_type = response.content.strip().lower()

                # Validate agent type
                if agent_type not in [
                    "insurance",
                    "quote_generator",
                    "registration",
                    "claims",
                    "billing",
                    "complaint",
                    "document_intelligence",
                    "general",
                ]:
                    agent_type = "general"

                # Check if agent is available
                if agent_type != "general" and agent_type not in agents:
                    agent_type = "general"

                # Be conservative: only route to specialized agent if intent is very clear
                # Otherwise, default to default agent
                if agent_type == "general":
                    logger.debug(f"Keeping message with default agent")
                    # Don't route, just return state indicating to stay with default
                    return {
                        **state,
                        "current_agent": "default",
                        "agent_type": "default",
                    }

                logger.info(f"Routing to specialized agent: {agent_type}")

                return {**state, "current_agent": agent_type, "agent_type": agent_type}

            except Exception as e:
                logger.error(f"Error in router node: {e}", exc_info=True)
                return {**state, "current_agent": "default", "agent_type": "default"}

        return router_node

    @staticmethod
    def _create_agent_node(
        agent_name: str, agent: Any, allowed_agents: Optional[List[str]] = None
    ):
        """Create a node for a specific agent.
        allowed_agents is used by the document_intelligence branch to validate suggested_next_agent.
        """
        _allowed_agents = allowed_agents or []

        async def agent_node(state: GraphState) -> GraphState:
            """Process message through the agent."""
            logger.info(f"[GRAPH_FACTORY] agent_node() called for agent: {agent_name}")

            if not agent:
                logger.warning(f"[GRAPH_FACTORY] Agent {agent_name} not available")
                return {**state, "agent_response": "Agent not available."}

            messages = state["messages"]
            last_message = messages[-1] if messages else None

            if not last_message or not isinstance(last_message, HumanMessage):
                logger.warning(f"[GRAPH_FACTORY] No valid message to process")
                return {**state, "agent_response": "No message to process."}

            try:
                # Handle document intelligence agent differently (it expects documents parameter)
                if agent_name == "document_intelligence":
                    logger.info(
                        f"[GRAPH_FACTORY] Entering document_intelligence agent node"
                    )

                    # Extract documents from context if available
                    context = state.get("context", {})
                    documents = context.get("documents", [])

                    logger.info(
                        f"[GRAPH_FACTORY] Calling document_intelligence agent with {len(documents)} document(s). Context keys: {list(context.keys()) if context else 'None'}"
                    )
                    if documents:
                        logger.info(
                            f"[GRAPH_FACTORY] Document details: {[{'id': d.get('id'), 'downloadUrl': d.get('downloadUrl'), 'cloudinaryUrl': d.get('cloudinaryUrl')} for d in documents]}"
                        )
                    else:
                        logger.warning(
                            f"[GRAPH_FACTORY] No documents found in context!"
                        )

                    # Build conversation history
                    conversation_history = [
                        {
                            "role": (
                                "user" if isinstance(m, HumanMessage) else "assistant"
                            ),
                            "content": m.content,
                        }
                        for m in messages[:-1]
                    ]

                    logger.info(
                        f"[GRAPH_FACTORY] About to call agent.process() with message: '{last_message.content[:100]}...'"
                    )

                    # Document intelligence agent uses async process method
                    result = await agent.process(
                        last_message.content,
                        conversation_history,
                        documents,
                    )

                    logger.info(
                        f"[GRAPH_FACTORY] agent.process() returned: {result.get('agent_type', 'unknown')}"
                    )

                    response = result.get(
                        "agent_response", "I'm processing your document request."
                    )
                    agent_type = result.get("agent_type", agent_name)

                    # If document intelligence suggested a downstream agent, set cross-sell and context
                    suggested = (result.get("suggested_next_agent") or "").strip().lower()
                    document_analysis = result.get("document_analysis") or response
                    new_context = dict(state.get("context", {}))
                    cross_sell_update = {}

                    if (
                        suggested
                        and suggested != "none"
                        and (not _allowed_agents or suggested in _allowed_agents)
                    ):
                        cross_sell_update = {
                            "cross_sell_opportunity": {
                                "detected": True,
                                "to_agent": suggested,
                                "from_agent": "document_intelligence",
                                "reason": f"Document classified as {suggested}",
                                "document_analysis": document_analysis,
                            },
                        }
                        new_context["document_analysis"] = document_analysis
                        logger.info(
                            f"[GRAPH_FACTORY] Document intelligence suggested handoff to: {suggested}"
                        )

                    return {
                        **state,
                        "agent_response": response,
                        "current_agent": agent_name,
                        "agent_type": agent_type,
                        "previous_agent": state.get("current_agent", ""),
                        "context": new_context,
                        **cross_sell_update,
                    }
                else:
                    ctx = dict(state.get("context", {}))
                    cross_sell = state.get("cross_sell_opportunity", {})
                    if not ctx.get("document_analysis") and cross_sell.get("from_agent") == "document_intelligence" and cross_sell.get("document_analysis"):
                        ctx["document_analysis"] = cross_sell["document_analysis"]
                    input_data = {
                        "message": last_message.content,
                        "conversation_id": state["conversation_id"],
                        "context": ctx,
                        "history": [
                            {
                                "role": (
                                    "user"
                                    if isinstance(m, HumanMessage)
                                    else "assistant"
                                ),
                                "content": m.content,
                            }
                            for m in messages[:-1]
                        ],
                    }

                    result = await agent.process(input_data)
                    response = result.get(
                        "content",
                        result.get("response", "I'm processing your request."),
                    )

                    # Extract agent_type from agent's response (for debugging/display)
                    agent_type = result.get("agent_type", agent_name)

                # Store previous agent for cross-sell detection
                previous_agent = state.get("current_agent", "")

                return {
                    **state,
                    "agent_response": response,
                    "current_agent": agent_name,
                    "agent_type": agent_type,
                    "previous_agent": previous_agent,
                }

            except Exception as e:
                logger.error(f"Error in {agent_name} agent node: {e}", exc_info=True)
                return {
                    **state,
                    "agent_response": f"Error processing request: {str(e)}",
                }

        return agent_node

    @staticmethod
    def _create_default_agent_node(
        llm: ChatOpenAI, tenant_id: str, tenant_config: Dict[str, Any]
    ):
        """
        Create the default agent node for tenant conversations.

        This is the tenant-specific default agent that handles most conversations. It will:
        - Handle general queries and conversations
        - Continue conversations naturally
        - Only hand off to specialized agents when cross-sell detector identifies opportunities
        """

        async def default_agent_node(state: GraphState) -> GraphState:
            """Handle conversations with the default agent."""
            messages = state["messages"]
            context = state.get("context", {})
            tenant_name = tenant_config.get("tenant_name", tenant_id)
            allowed_agents = tenant_config.get("allowed_agents", [])
            allowed_tools = tenant_config.get("allowed_tools", [])

            # Check if documents are present
            documents = context.get("documents", [])
            if documents:
                logger.info(f"Default agent detected {len(documents)} document(s)")

                # Check if document_intelligence agent is available
                # Don't route if documents have already been processed
                has_doc_intelligence = "document_intelligence" in allowed_agents
                documents_already_processed = state.get("documents_processed", False)

                if has_doc_intelligence and not documents_already_processed:
                    logger.info("Routing to document_intelligence agent")
                    # Set cross-sell opportunity so cross-sell detector routes to document_intelligence
                    return {
                        **state,
                        "current_agent": "default",
                        "agent_type": "default",
                        "agent_response": "",  # Clear response so document_intelligence can provide its own
                        "cross_sell_opportunity": {
                            "detected": True,
                            "to_agent": "document_intelligence",
                            "from_agent": "default",
                            "reason": "Documents detected in message",
                        },
                    }
                elif documents_already_processed:
                    logger.info(
                        "Documents already processed, skipping document_intelligence routing"
                    )
                else:
                    # Document intelligence not available, but still inform user about documents
                    logger.warning(
                        f"Documents detected but document_intelligence agent not enabled for tenant {tenant_id}"
                    )
                    doc_list = ", ".join(
                        [d.get("fileName", "Unknown") for d in documents]
                    )
                    return {
                        **state,
                        "agent_response": f"I see you've attached {len(documents)} document(s): {doc_list}. To process these documents, please enable the Document Intelligence agent for your tenant.",
                        "agent_type": "default",
                    }

            # Founder doctrine / brand principles: give the agent context so it can answer using them
            doctrine_blob = ""
            try:
                principles = get_principles()
                if principles:
                    doctrine_blob = "\n\nFounder doctrine and brand principles (use these when users ask about brand, values, strategy, or how we operate):\n"
                    for p in principles:
                        name = p.get("name", "")
                        rules = p.get("rules") or []
                        doctrine_blob += f"- {name}: " + "; ".join(rules) + "\n"
                    doctrine_blob += "When users ask for advice based on 'your brand', 'your values', or strategy, base your answer on these principles. Do not ask them to tell you about the brand—you already have it above."
            except Exception as e:
                logger.debug("Could not load doctrine for default agent prompt: %s", e)

            system_prompt = f"""You are the default AI assistant for {tenant_name}.
You handle general conversations and help users with their questions and tasks.

Your role:
- Provide helpful, accurate, and respectful responses
- Continue conversations naturally
- Help with general inquiries and questions
- If a user needs specialized help (insurance quotes, claims, billing, etc.), 
  you can acknowledge their need and the system will route them appropriately

Available specialized agents: {', '.join(allowed_agents) if allowed_agents else 'None configured'}
Available tools: {', '.join(allowed_tools) if allowed_tools else 'None configured'}
{doctrine_blob}

Escalation to human: If the user asks to speak to a human, wants a real person, wants to be transferred to an agent, or you cannot resolve their issue and they need human help, you MUST start your reply with exactly this line on its own first line: [ESCALATE]
Then on the next line(s) give a short, reassuring message that a team member will be with them shortly. Example:
[ESCALATE]
I'm connecting you with our team. A team member will be with you shortly.

Be conversational, friendly, and helpful. If you cannot help with something specific, 
politely explain and suggest alternatives. Use [ESCALATE] only when the user clearly wants human support or you cannot help further."""

            try:
                # Check if user message contains claims-related keywords
                last_user_message = None
                last_assistant_message = None
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage) and last_user_message is None:
                        last_user_message = (msg.content or "").strip().lower()
                        if last_assistant_message is not None:
                            break
                    elif isinstance(msg, AIMessage) and last_assistant_message is None:
                        last_assistant_message = (msg.content or "").strip().lower()
                        if last_user_message is not None:
                            break

                # If last assistant message was from a claims flow and user replied with affirmation, route to claims
                if (
                    last_user_message is not None
                    and last_assistant_message is not None
                    and "claims" in allowed_agents
                ):
                    claims_ctx = any(
                        p in last_assistant_message
                        for p in (
                            "file this claim",
                            "would you like me to file",
                            "file a claim",
                            "policy number",
                            "incident date",
                            "claim for you",
                            "submit this claim",
                        )
                    )
                    affirmation = (
                        last_user_message in ("yes", "yes please", "please", "sure", "ok", "okay", "go ahead", "please do", "do it", "yep", "yeah")
                        or any(last_user_message.startswith(a) for a in ["yes ", "yes,", "sure,", "ok ", "ok,"])
                        or (len(last_user_message) <= 20 and "yes" in last_user_message)
                    )
                    if claims_ctx and affirmation:
                        logger.info(
                            "Default agent: user continuing claims conversation with affirmation, routing to claims"
                        )
                        return {
                            **state,
                            "current_agent": "default",
                            "agent_type": "default",
                            "agent_response": "",
                            "cross_sell_opportunity": {
                                "detected": True,
                                "to_agent": "claims",
                                "from_agent": "default",
                                "reason": "User affirming claims flow (e.g. yes to filing claim)",
                            },
                        }

                # Detect claims-related queries and set up cross-sell opportunity
                if last_user_message and "claims" in allowed_agents:
                    claims_keywords = [
                        "claim",
                        "claims",
                        "claim status",
                        "file a claim",
                        "check my claim",
                        "my claim",
                        "claim number",
                        "claim details",
                        "claim information",
                        "submit a claim",
                        "new claim",
                    ]
                    if any(keyword in last_user_message for keyword in claims_keywords):
                        logger.info(
                            "Default agent detected claims-related query, setting up cross-sell"
                        )
                        return {
                            **state,
                            "current_agent": "default",
                            "agent_type": "default",
                            "agent_response": "",  # Clear response so claims agent can provide its own
                            "cross_sell_opportunity": {
                                "detected": True,
                                "to_agent": "claims",
                                "from_agent": "default",
                                "reason": "Claims-related query detected",
                            },
                        }

                # Build message list with system prompt
                llm_messages = [SystemMessage(content=system_prompt)] + messages
                response = await llm.ainvoke(llm_messages)
                response_content = response.content

                return {
                    **state,
                    "agent_response": response_content,
                    "current_agent": "default",
                    "agent_type": "default",
                }

            except Exception as e:
                logger.error(
                    f"Error in default agent node for tenant {tenant_id}: {e}",
                    exc_info=True,
                )
                return {
                    **state,
                    "agent_response": "I apologize, but I encountered an error. Please try again.",
                    "agent_type": "default",
                }

        return default_agent_node

    @staticmethod
    def _create_tool_executor_node(tool_wrapper: ToolWrapper):
        """Create a node for executing tools."""

        async def tool_executor_node(state: GraphState) -> GraphState:
            """Execute tools if needed."""
            tool_calls = state.get("tool_calls", [])

            if not tool_calls:
                return {
                    **state,
                    "agent_response": "No tools to execute.",
                    "agent_type": "tool_executor",
                }

            results = []
            for tool_call in tool_calls:
                try:
                    result = await tool_wrapper.call_tool(
                        tool_call["tool_name"], tool_call.get("parameters", {})
                    )
                    results.append({"tool": tool_call["tool_name"], "result": result})
                except Exception as e:
                    logger.error(
                        f"Error executing tool {tool_call.get('tool_name')}: {e}"
                    )
                    results.append(
                        {"tool": tool_call.get("tool_name"), "error": str(e)}
                    )

            return {
                **state,
                "agent_response": f"Tool execution completed: {results}",
                "agent_type": "tool_executor",
            }

        return tool_executor_node

    @staticmethod
    def _create_cross_sell_detector_node(llm: ChatOpenAI, agents: Dict[str, Any]):
        """Create a cross-sell detection node that identifies opportunities to route to other agents."""

        async def cross_sell_detector_node(state: GraphState) -> GraphState:
            """
            Detect cross-sell opportunities and determine if routing to another agent is appropriate.

            This node analyzes the conversation context and current agent response to identify
            if the user's query or intent would be better served by a different agent, enabling
            seamless cross-domain handoffs.
            """
            current_agent = state.get("current_agent", "general")
            agent_response = state.get("agent_response", "")
            messages = state.get("messages", [])
            previous_agent = state.get("previous_agent", "")
            cross_sell = state.get("cross_sell_opportunity", {})

            # FIRST: If document_intelligence just processed, either preserve handoff or go to final
            # This check must come BEFORE preserving existing cross-sell opportunities
            if current_agent == "document_intelligence" and agent_response:
                cross_sell_to = cross_sell.get("to_agent")
                if cross_sell_to:
                    # Document intelligence set a suggested_next_agent; preserve it for routing
                    logger.info(
                        f"[CROSS_SELL_DETECTOR] Document intelligence suggested handoff to {cross_sell_to}; preserving."
                    )
                    return {
                        **state,
                        "documents_processed": True,
                    }
                logger.info(
                    f"[CROSS_SELL_DETECTOR] Document intelligence has processed documents; no suggested agent. Routing to final."
                )
                return {
                    **state,
                    "cross_sell_opportunity": {},
                    "documents_processed": True,
                }

            # Clear cross-sell if claims agent has just processed (prevent infinite loop)
            if current_agent == "claims" and agent_response:
                logger.info(
                    f"[CROSS_SELL_DETECTOR] Claims agent has processed the request. Clearing cross-sell to prevent loop."
                )
                return {
                    **state,
                    "cross_sell_opportunity": {},  # Clear cross-sell
                    "current_agent": "final",  # Explicitly set to final
                }

            # SECOND: Check if a cross-sell opportunity was already set (e.g., by default agent for documents)
            # Only preserve if we haven't already processed documents or claims
            if cross_sell.get("detected") and cross_sell.get("to_agent"):
                # Don't preserve if documents have already been processed
                if state.get("documents_processed", False):
                    logger.info(
                        f"[CROSS_SELL_DETECTOR] Documents already processed, clearing cross-sell opportunity to prevent loop"
                    )
                    return {
                        **state,
                        "cross_sell_opportunity": {},
                    }
                # Don't preserve if we just processed with the target agent (prevent loop)
                target_agent = cross_sell.get("to_agent")
                if current_agent == target_agent and agent_response:
                    logger.info(
                        f"[CROSS_SELL_DETECTOR] Already processed by {target_agent} agent, clearing cross-sell to prevent loop"
                    )
                    return {
                        **state,
                        "cross_sell_opportunity": {},
                    }
                # A cross-sell opportunity was already set, preserve it
                logger.info(
                    f"[CROSS_SELL_DETECTOR] Preserving existing cross-sell opportunity: {cross_sell.get('from_agent')} → {cross_sell.get('to_agent')}"
                )
                return (
                    state  # Return state as-is, preserving the cross-sell opportunity
                )

            # Don't cross-sell if we just came from another agent via cross-sell (prevent infinite loops)
            # But allow if this is the first agent processing the message
            if (
                cross_sell.get("detected")
                and previous_agent
                and previous_agent != current_agent
            ):
                logger.debug(
                    f"Skipping cross-sell detection - already cross-sold from {previous_agent} to {current_agent}"
                )
                return {**state, "cross_sell_opportunity": {}}

            # Get the last user message for context
            last_user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break

            if not last_user_message:
                return {**state, "cross_sell_opportunity": {}}

            # Build cross-sell detection prompt
            available_agents = list(agents.keys())
            cross_sell_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""You are a cross-sell detection assistant. Analyze the conversation to identify if the user's query or intent would be better served by a different specialized agent.

Current agent: {current_agent}
Available agents: {', '.join(available_agents) if available_agents else 'None'}

User's last message: "{last_user_message}"
Current agent's response: "{agent_response[:200]}..." (truncated)

Cross-sell opportunities to look for:
- User explicitly asking about insurance policies/coverage → route to "insurance"
- User explicitly asking for quotes/pricing → route to "quote_generator"
- User explicitly asking about claims, claim status, filing a claim, checking claims, or any claim-related query → route to "claims"
- User explicitly asking about billing, payments, invoices → route to "billing"
- User explicitly asking about billing/payments → route to "billing"
- User explicitly asking about registration/onboarding → route to "registration"
- User explicitly complaining or reporting issues → route to "complaint"

Rules:
1. Only suggest cross-sell if there's a VERY CLEAR and EXPLICIT opportunity
2. The default agent (general) should handle most conversations naturally
3. Only route to specialized agents when the user explicitly needs that specific service
4. Don't cross-sell if the current agent already addressed the query adequately
5. Don't cross-sell if we just routed from another agent (to prevent loops)
6. If no clear cross-sell opportunity, respond with "none"

Respond with ONLY one of: insurance, quote_generator, registration, claims, billing, complaint, general, or none""",
                    ),
                    (
                        "human",
                        "Analyze this conversation and determine if a cross-sell opportunity exists.",
                    ),
                ]
            )

            try:
                chain = cross_sell_prompt | llm
                response = await chain.ainvoke({})
                suggested_agent = response.content.strip().lower()

                # Validate and check if agent is available
                if suggested_agent == "none" or suggested_agent == current_agent:
                    logger.debug(
                        f"No cross-sell opportunity detected (suggested: {suggested_agent}, current: {current_agent})"
                    )
                    return {**state, "cross_sell_opportunity": {}}

                if (
                    suggested_agent not in available_agents
                    and suggested_agent != "general"
                ):
                    logger.debug(
                        f"Suggested cross-sell agent {suggested_agent} not available"
                    )
                    return {**state, "cross_sell_opportunity": {}}

                # Cross-sell opportunity detected
                logger.info(
                    f"Cross-sell opportunity detected: {current_agent} → {suggested_agent} "
                    f"(conversation: {state.get('conversation_id', 'unknown')})"
                )

                # When routing to another agent, preserve the conversation context
                # The new agent will receive the full message history and can provide
                # a seamless response that acknowledges the transition
                return {
                    **state,
                    "cross_sell_opportunity": {
                        "detected": True,
                        "from_agent": current_agent,
                        "to_agent": suggested_agent,
                        "reason": f"User query suggests {suggested_agent} would be more appropriate",
                    },
                    "previous_agent": current_agent,
                    # Clear current agent response so the new agent can provide its own
                    # This allows the new agent to naturally incorporate context
                    "agent_response": "",
                }

            except Exception as e:
                logger.error(f"Error in cross-sell detector node: {e}", exc_info=True)
                # On error, proceed to final response
                return {**state, "cross_sell_opportunity": {}}

        return cross_sell_detector_node

    @staticmethod
    def _route_from_cross_sell(state: GraphState) -> str:
        """
        Route from cross-sell detector to either another agent or final response.

        If a cross-sell opportunity was detected, route to the suggested agent.
        Otherwise, route to final response.

        This enables seamless handoff between agents without rigid domain boundaries,
        allowing the system to recognize opportunities and guide users naturally.
        Security boundaries are maintained - sensitive data stays siloed per tenant.
        """
        cross_sell = state.get("cross_sell_opportunity", {})
        documents_processed = state.get("documents_processed", False)

        logger.info(
            f"[ROUTE_FROM_CROSS_SELL] cross_sell_opportunity: {cross_sell}, detected: {cross_sell.get('detected')}, to_agent: {cross_sell.get('to_agent')}, documents_processed: {documents_processed}"
        )

        # Don't route to document_intelligence if documents are already processed
        if cross_sell.get("detected") and cross_sell.get("to_agent"):
            target_agent = cross_sell["to_agent"]
            from_agent = cross_sell.get("from_agent", "unknown")
            current_agent = state.get("current_agent", "")

            # Prevent routing back to document_intelligence if already processed
            if target_agent == "document_intelligence" and documents_processed:
                logger.info(
                    f"[ROUTE_FROM_CROSS_SELL] Documents already processed, routing to final instead of {target_agent}"
                )
                return "final"

            # Prevent routing back to claims if we just came from claims (prevent infinite loop)
            if target_agent == "claims" and current_agent == "claims":
                logger.info(
                    f"[ROUTE_FROM_CROSS_SELL] Already processed by claims agent, routing to final to prevent loop"
                )
                return "final"

            logger.info(
                f"[ROUTE_FROM_CROSS_SELL] Routing cross-sell handoff: {from_agent} → {target_agent} "
                f"(conversation: {state.get('conversation_id', 'unknown')})"
            )
            return target_agent

        logger.info(
            f"[ROUTE_FROM_CROSS_SELL] No cross-sell opportunity, routing to final"
        )
        return "final"

    @staticmethod
    def _create_doctrine_evaluation_node(legacy_agent: LegacyAgent):
        """Create the doctrine evaluation node: router -> fast engine or LegacyAgent -> state update."""

        async def doctrine_evaluation_node(state: GraphState) -> GraphState:
            agent_response = state.get("agent_response", "")
            agent_type = state.get("agent_type", "default")
            context = dict(state.get("context", {}))
            context["agent_response"] = agent_response
            # Include last user message so router can detect strategic intent (e.g. "strategy memo for the board")
            messages = state.get("messages") or []
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    context["user_message"] = getattr(m, "content", "") or ""
                    break

            document_id = None
            if context.get("documents"):
                doc = context["documents"][0] if context["documents"] else None
                if isinstance(doc, dict) and doc.get("id"):
                    document_id = doc.get("id")

            t0 = time.perf_counter()
            route_decision = doctrine_router_route(context, agent_type)
            route_ms = int((time.perf_counter() - t0) * 1000)
            emit_doctrine_event(
                "doctrine_router.route",
                {
                    "document_id": document_id,
                    "decision_type": route_decision,
                    "latency_ms": route_ms,
                    "agent_type": agent_type,
                    "conversation_id": state.get("conversation_id"),
                },
            )

            try:
                if route_decision == "FAST_RULES":
                    t1 = time.perf_counter()
                    result = doctrine_engine_evaluate(agent_response, context)
                    eval_ms = int((time.perf_counter() - t1) * 1000)
                    emit_doctrine_event(
                        "doctrine_alignment.scored",
                        {
                            "document_id": document_id,
                            "alignment_score": result.institutional_alignment_score,
                            "decision_type": route_decision,
                            "latency_ms": eval_ms,
                        },
                    )
                else:
                    t1 = time.perf_counter()
                    emit_doctrine_event(
                        "legacy_agent.invoked",
                        {
                            "document_id": document_id,
                            "decision_type": route_decision,
                            "agent_type": agent_type,
                        },
                    )
                    result = await legacy_agent.evaluate(agent_response, context)
                    eval_ms = int((time.perf_counter() - t1) * 1000)
                    emit_doctrine_event(
                        "doctrine_alignment.scored",
                        {
                            "document_id": document_id,
                            "alignment_score": result.institutional_alignment_score,
                            "decision_type": route_decision,
                            "latency_ms": eval_ms,
                        },
                    )
            except Exception as e:
                logger.exception("Doctrine evaluation failed: %s", e)
                result = empty_evaluation()

            doctrine_evaluation_dict = result.model_dump()
            return {
                **state,
                "doctrine_evaluation": doctrine_evaluation_dict,
                "doctrine_route": route_decision,
            }

        return doctrine_evaluation_node

    @staticmethod
    def _create_final_response_node():
        """Create the final response node."""

        async def final_response_node(state: GraphState) -> GraphState:
            """
            Prepare final response.

            If a cross-sell occurred, the agent response will naturally incorporate
            the context from the previous agent, creating a seamless experience.
            The conversation history is preserved, so the new agent can reference
            previous interactions naturally.
            """
            agent_response = state.get("agent_response", "I'm here to help.")

            # Log cross-sell completion for analytics and monitoring
            cross_sell = state.get("cross_sell_opportunity", {})
            if cross_sell.get("detected"):
                from_agent = cross_sell.get("from_agent", "")
                to_agent = cross_sell.get("to_agent", "")
                logger.info(
                    f"Finalizing response after cross-sell: {from_agent} → {to_agent} "
                    f"(conversation: {state.get('conversation_id', 'unknown')})"
                )

            # Add assistant message to state
            new_messages = state["messages"] + [AIMessage(content=agent_response)]

            return {**state, "messages": new_messages}

        return final_response_node

    @staticmethod
    def _route_to_agent(state: GraphState) -> str:
        """Route to the appropriate agent based on current_agent."""
        current_agent = state.get("current_agent", "general")

        # Check if tools are needed (simplified - in production, check for tool calls)
        if state.get("tool_calls"):
            return "tools"

        return current_agent
