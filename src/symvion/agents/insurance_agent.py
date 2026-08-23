"""
Insurance agent - Root orchestrator for insurance matters.
Coordinates insurance-related workflows and routes to specialized agents.
"""
import os
import logging
from typing import Dict, Any, List, Optional
from symvion.providers.factory import ProviderFactory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class InsuranceAgent:
    """Root orchestrator agent for insurance-related matters."""

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        """
        Initialize insurance agent.

        Args:
            tenant_id: Unique tenant identifier
            config: Agent-specific configuration including sub-agents
        """
        self.tenant_id = tenant_id
        self.config = config
        self.description = "General insurance policy manager. Handles policy lookups, basic coverage questions, and account history."
        
        # Initialize LLM
        self.llm = ProviderFactory.get_provider(
            config.get("provider", "openai"),
            config
        )
        
        # Available sub-agents for insurance matters
        self.available_agents = config.get("available_agents", [
            "claims", "billing", "quote_generator", "registration"
        ])
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for insurance agent."""
        return f"""You are an intelligent insurance orchestrator and advisor for tenant {self.tenant_id}.

Your role is to:
- Serve as the primary point of contact for all insurance-related inquiries
- Understand customer needs and route them to appropriate specialized agents when needed
- Provide comprehensive insurance advice and guidance
- Coordinate between different insurance services (claims, billing, quotes, policies)
- Help customers understand their insurance coverage, policies, and options
- Answer general insurance questions about coverage types, premiums, deductibles, etc.

Available specialized agents: {', '.join(self.available_agents)}

When a customer asks about:
- Insurance claims → Guide them to the claims process or provide claims information
- Billing or payments → Help with billing inquiries or route to billing support
- Getting a quote → Assist with quote generation or route to quote specialist
- Policy questions → Provide information about coverage, terms, and conditions
- General insurance → Provide comprehensive insurance advice

Be knowledgeable, professional, and helpful. If a query requires specialized handling, acknowledge it and either provide the information yourself or guide the customer to the appropriate specialist.

Always maintain a customer-focused approach and ensure clear communication about insurance matters."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process insurance-related input and coordinate with sub-agents if needed.

        Args:
            input_data: Input data containing:
                - message: User's message content
                - conversation_id: Conversation identifier
                - context: Additional context
                - history: Previous conversation messages

        Returns:
            Dict with 'content' or 'response' key containing the agent's response
        """
        try:
            message = input_data.get("message", "")
            history = input_data.get("history", [])
            context = input_data.get("context", {})
            
            # Build conversation history
            messages = [SystemMessage(content=self.system_prompt)]
            
            # Add conversation history
            for hist_msg in history[-10:]:  # Last 10 messages for context
                if isinstance(hist_msg, dict):
                    role = hist_msg.get("role")
                    content = hist_msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                elif isinstance(hist_msg, HumanMessage):
                    messages.append(hist_msg)
                elif isinstance(hist_msg, AIMessage):
                    messages.append(hist_msg)
                elif isinstance(hist_msg, SystemMessage):
                    messages.append(hist_msg)
            
            # Add current user message
            messages.append(HumanMessage(content=message))
            
            # Get response from LLM
            final_response = await self.llm.ainvoke(messages)
            response_content = final_response.content
            usage = final_response.usage_metadata if hasattr(final_response, "usage_metadata") else getattr(final_response, "response_metadata", {}).get("token_usage", {})
            
            logger.debug(f"Insurance agent response for tenant {self.tenant_id}: {response_content[:100]}")
            
            return {
                "content": response_content,
                "agent_type": "insurance",
                "tenant_id": self.tenant_id,
                "usage_metadata": usage
            }
            
        except Exception as e:
            logger.error(f"Error in insurance agent: {e}", exc_info=True)
            return {
                "content": "I apologize, but I encountered an error while processing your insurance inquiry. Please try again or contact support.",
                "agent_type": "insurance",
                "error": str(e),
            }
