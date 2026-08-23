"""
Tenant-scoped memory store.
Manages conversation memory with strict tenant isolation.
"""
from typing import Dict, Any, Optional


class MemoryStore:
    """Tenant-scoped memory store for conversation history."""

    def __init__(self, tenant_id: str):
        """
        Initialize memory store for a tenant.

        Args:
            tenant_id: Unique tenant identifier
        """
        self.tenant_id = tenant_id
        self._memory: Dict[str, Any] = {}

    def get_conversation_memory(
        self, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve memory for a specific conversation.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            Conversation memory or None if not found
        """
        return self._memory.get(conversation_id)

    def update_conversation_memory(
        self, conversation_id: str, memory: Dict[str, Any]
    ):
        """
        Update memory for a specific conversation.

        Args:
            conversation_id: Unique conversation identifier
            memory: Memory data to store
        """
        self._memory[conversation_id] = memory

    def clear_conversation_memory(self, conversation_id: str):
        """
        Clear memory for a specific conversation.

        Args:
            conversation_id: Unique conversation identifier
        """
        if conversation_id in self._memory:
            del self._memory[conversation_id]
