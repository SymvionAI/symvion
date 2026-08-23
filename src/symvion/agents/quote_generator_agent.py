"""
Quote generator agent.
Handles insurance quote generation and pricing inquiries.
"""
import os
import logging
from typing import Dict, Any
from symvion.providers.factory import ProviderFactory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class QuoteGeneratorAgent:
    """Agent for generating insurance quotes."""

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        """
        Initialize quote generator agent.

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
        """Build the system prompt for quote generator agent."""
        return f"""You are a professional insurance quote generator assistant for tenant {self.tenant_id}.

Your role is to:
- Help customers get insurance quotes for various coverage types
- Collect necessary information for accurate quote generation
- Explain different coverage options and their pricing
- Guide customers through the quote process step by step
- Answer questions about premiums, deductibles, coverage limits, and policy terms
- Compare different insurance options and explain the differences
- Provide preliminary estimates based on customer information

When generating quotes, you should:
1. Ask for essential information:
   - Type of insurance needed (auto, home, health, life, etc.)
   - Coverage requirements and limits
   - Personal/business information relevant to the quote
   - Any specific preferences or requirements

2. Explain the quote process clearly:
   - What information is needed and why
   - How quotes are calculated
   - What factors affect pricing
   - Timeline for receiving a final quote

3. Be transparent about:
   - What is included in the quote
   - What additional information might be needed
   - Next steps after receiving a quote
   - How to proceed with purchasing coverage

Be professional, thorough, and helpful. Always confirm important details before providing quote information.
If you need specific information that the customer hasn't provided, ask for it clearly and explain why it's needed."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process quote generation requests.

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
            
            logger.debug(f"Quote generator agent response for tenant {self.tenant_id}: {response_content[:100]}")
            
            return {
                "content": response_content,
                "agent_type": "quote_generator",
                "tenant_id": self.tenant_id,
            }
            
        except Exception as e:
            logger.error(f"Error in quote generator agent: {e}", exc_info=True)
            return {
                "content": "I apologize, but I encountered an error while processing your quote request. Please try again or contact support.",
                "agent_type": "quote_generator",
                "error": str(e),
            }
