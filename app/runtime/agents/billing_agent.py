"""
Billing agent.
Handles billing inquiries and payment processing workflows.
"""
import os
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class BillingAgent:
    """Agent for handling billing operations."""

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        """
        Initialize billing agent.

        Args:
            tenant_id: Unique tenant identifier
            config: Agent-specific configuration
        """
        self.tenant_id = tenant_id
        self.config = config
        
        # Initialize LLM
        model_name = config.get("model", os.getenv("LLM_MODEL", "gpt-4"))
        temperature = config.get("temperature", float(os.getenv("LLM_TEMPERATURE", "0.7")))
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for billing agent."""
        return f"""You are a helpful billing and payment assistant for tenant {self.tenant_id}.

Your role is to:
- Answer questions about billing statements and invoices
- Help with payment processing and payment methods
- Explain billing charges and fees
- Assist with account balance inquiries
- Guide customers through payment procedures
- Help resolve billing disputes and questions
- Explain payment terms, due dates, and late fees

Be professional, clear, and helpful. When dealing with billing:
- Ask for account or invoice numbers when needed
- Provide clear explanations of charges
- Guide customers on payment options and methods
- Explain billing cycles and payment schedules
- If you don't have access to specific account data, guide the customer on how to access it

Always be transparent about billing information and help customers understand their statements."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process billing-related input.

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
            
            logger.debug(f"Billing agent response for tenant {self.tenant_id}: {response_content[:100]}")
            
            return {
                "content": response_content,
                "agent_type": "billing",
                "tenant_id": self.tenant_id,
            }
            
        except Exception as e:
            logger.error(f"Error in billing agent: {e}", exc_info=True)
            return {
                "content": "I apologize, but I encountered an error while processing your billing inquiry. Please try again or contact support.",
                "agent_type": "billing",
                "error": str(e),
            }
