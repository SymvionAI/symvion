"""
Memory schemas for conversation and agent state.
Defines data structures for memory storage.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime


class ConversationMemory:
    """Schema for conversation memory."""

    def __init__(
        self,
        conversation_id: str,
        tenant_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.conversation_id = conversation_id
        self.tenant_id = tenant_id
        self.messages = messages
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
