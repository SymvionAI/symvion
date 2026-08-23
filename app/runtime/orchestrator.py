"""
Root orchestrator agent that coordinates tenant-specific workflows.
Handles routing, agent selection, and tool execution.
"""

import os
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import logging

from app.runtime.memory.memory_store import MemoryStore
from app.utils.logging import log_to_langsmith

logger = logging.getLogger(__name__)


class Orchestrator:
    """Root orchestrator for tenant AI workflows."""

    def __init__(
        self,
        tenant_id: str,
        config: Dict[str, Any],
        memory_store: Optional[MemoryStore] = None,
    ):
        """
        Initialize orchestrator for a tenant.

        Args:
            tenant_id: Unique tenant identifier
            config: Tenant configuration including allowed agents and tools
            memory_store: Optional memory store for conversation history
        """
        self.tenant_id = tenant_id
        self.config = config
        self.tenant_name = (
            config.get("tenant_name") or tenant_id
        )  # Use tenant_name if available, fallback to tenant_id
        self.allowed_agents = config.get("allowed_agents", [])
        self.allowed_tools = config.get("allowed_tools", [])
        self.memory_store = memory_store or MemoryStore(tenant_id)

        # Initialize LLM
        model_name = os.getenv("LLM_MODEL", "gpt-4")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

        # System prompt for the orchestrator
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on tenant configuration."""
        prompt = f"""You are an AI assistant for {self.tenant_name}.
You help users with their questions and tasks in a helpful, accurate, and safe manner.

Allowed agents: {', '.join(self.allowed_agents) if self.allowed_agents else 'All'}
Allowed tools: {', '.join(self.allowed_tools) if self.allowed_tools else 'All'}

Always be helpful, accurate, and respectful. If you cannot help with something, 
politely explain why and suggest alternatives if possible."""
        return prompt

    async def process_message(
        self, message: str, conversation_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an incoming message through the orchestrator.

        Args:
            message: User message content
            conversation_id: Unique conversation identifier
            context: Additional context for the conversation

        Returns:
            Response dictionary with agent output and metadata
        """
        try:
            # Retrieve conversation memory
            conversation_memory = self.memory_store.get_conversation_memory(
                conversation_id
            )
            conversation_history = (
                conversation_memory.get("history", []) if conversation_memory else []
            )

            # Build messages for LLM
            messages = [SystemMessage(content=self.system_prompt)]

            # Add conversation history
            for hist_msg in conversation_history[-10:]:  # Last 10 messages for context
                if hist_msg.get("role") == "user":
                    messages.append(HumanMessage(content=hist_msg.get("content", "")))
                elif hist_msg.get("role") == "assistant":
                    messages.append(SystemMessage(content=hist_msg.get("content", "")))

            # Add current user message
            messages.append(HumanMessage(content=message))

            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            response_content = response.content

            # Update conversation memory
            if not conversation_memory:
                conversation_memory = {"history": []}

            conversation_memory["history"].append({"role": "user", "content": message})
            conversation_memory["history"].append(
                {"role": "assistant", "content": response_content}
            )
            self.memory_store.update_conversation_memory(
                conversation_id, conversation_memory
            )

            # Log to LangSmith
            log_to_langsmith(
                tenant_id=self.tenant_id,
                conversation_id=conversation_id,
                agent_type="orchestrator",
                metadata={
                    "model": os.getenv("LLM_MODEL", "gpt-4"),
                    "message_length": len(message),
                    "response_length": len(response_content),
                },
            )

            return {
                "content": response_content,
                "metadata": {
                    "model": os.getenv("LLM_MODEL", "gpt-4"),
                    "tenant_id": self.tenant_id,
                    "conversation_id": conversation_id,
                    "agent_type": "orchestrator",  # For debugging - shows which agent handled the request
                },
            }

        except Exception as e:
            logger.error(
                f"Error processing message for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            # Return error response
            return {
                "content": "I apologize, but I encountered an error processing your message. Please try again.",
                "metadata": {
                    "error": str(e),
                    "tenant_id": self.tenant_id,
                    "conversation_id": conversation_id,
                    "agent_type": "orchestrator",
                },
            }
