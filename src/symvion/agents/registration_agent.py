"""
Registration agent.
Handles customer registration and onboarding workflows.
"""
import os
import logging
from typing import Dict, Any
from symvion.providers.factory import ProviderFactory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class RegistrationAgent:
    """Agent for handling customer registration and onboarding."""

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        """
        Initialize registration agent.

        Args:
            tenant_id: Unique tenant identifier
            config: Agent-specific configuration
        """
        self.tenant_id = tenant_id
        self.config = config
        
        # Initialize LLM
        self.llm = ProviderFactory.get_provider(
            config.get("provider", "openai"),
            config
        )
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for registration agent."""
        return f"""You are a helpful registration and onboarding assistant for tenant {self.tenant_id}.

Your role is to:
- Guide customers through the registration process
- Help with account creation and setup
- Answer questions about onboarding procedures
- Collect necessary information for account setup
- Provide clear, step-by-step instructions

Be friendly, professional, and patient. If you need specific information from the customer, ask for it clearly.
Always confirm important details before proceeding with registration steps."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process registration-related input.

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
                if hist_msg.get("role") == "user":
                    messages.append(HumanMessage(content=hist_msg.get("content", "")))
                elif hist_msg.get("role") == "assistant":
                    messages.append(AIMessage(content=hist_msg.get("content", "")))
            
            # Add current user message
            messages.append(HumanMessage(content=message))
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            response_content = response.content
            
            logger.debug(f"Registration agent response for tenant {self.tenant_id}: {response_content[:100]}")
            
            return {
                "content": response_content,
                "agent_type": "registration",
                "tenant_id": self.tenant_id,
            }
            
        except Exception as e:
            logger.error(f"Error in registration agent: {e}", exc_info=True)
            return {
                "content": "I apologize, but I encountered an error while processing your registration request. Please try again or contact support.",
                "agent_type": "registration",
                "error": str(e),
            }
