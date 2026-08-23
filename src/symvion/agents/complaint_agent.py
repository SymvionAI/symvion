"""
Complaint handling agent.
Handles customer complaints and escalation workflows.
"""

import os
import logging
from typing import Dict, Any
from symvion.providers.factory import ProviderFactory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class ComplaintAgent:
    """Agent for handling customer complaints."""

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        """
        Initialize complaint agent.

        Args:
            tenant_id: Unique tenant identifier
            config: Agent-specific configuration
        """
        self.tenant_id = tenant_id
        self.config = config
        self.description = "Resolves customer complaints, files tickets, and escalates to human agents."

        # Initialize LLM
        self.llm = ProviderFactory.get_provider(
            config.get("provider", "openai"),
            config
        )

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for complaint agent."""
        return f"""You are an empathetic and professional complaint resolution assistant for tenant {self.tenant_id}.

Your role is to:
- Listen actively to customer complaints and concerns
- Acknowledge the customer's frustration and validate their feelings
- Gather detailed information about the issue
- Provide clear explanations and potential solutions
- Escalate issues when appropriate
- Follow up on complaint resolution
- Help customers understand next steps and timelines

Be patient, understanding, and solution-oriented. When handling complaints:
- Always start by acknowledging the customer's concern
- Ask clarifying questions to fully understand the issue
- Provide realistic expectations about resolution timelines
- Offer multiple solutions when possible
- Document the complaint details clearly
- Explain escalation procedures if the issue cannot be resolved immediately

Remember: Your goal is to help resolve the issue while maintaining a positive customer relationship. Be empathetic but professional."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complaint-related input.

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
            
            logger.debug(f"Complaint agent response for tenant {self.tenant_id}: {response_content[:100]}")
            
            return {
                "content": response_content,
                "agent_type": "complaint",
                "tenant_id": self.tenant_id,
                "usage_metadata": usage
            }

        except Exception as e:
            logger.error(f"Error in complaint agent: {e}", exc_info=True)
            return {
                "content": "I apologize, but I encountered an error while processing your complaint. Please try again or contact support directly.",
                "agent_type": "complaint",
                "error": str(e),
            }
