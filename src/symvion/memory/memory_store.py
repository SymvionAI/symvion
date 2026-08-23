from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import logging
from langchain_core.messages import messages_to_dict, messages_from_dict
from symvion.config.models import MemoryConfig, MemoryStoreType

logger = logging.getLogger(__name__)

class BaseMemoryStore(ABC):
    """Base class for all memory stores."""
    
    @abstractmethod
    async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation memory."""
        pass
    
    @abstractmethod
    async def set(self, conversation_id: str, memory: Dict[str, Any], ttl: Optional[int] = None):
        """Update conversation memory."""
        pass
    
    @abstractmethod
    async def delete(self, conversation_id: str):
        """Clear conversation memory."""
        pass

class InProcessMemoryStore(BaseMemoryStore):
    """In-memory storage using a local dictionary."""
    
    def __init__(self):
        self._memory: Dict[str, Any] = {}

    async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._memory.get(conversation_id)

    async def set(self, conversation_id: str, memory: Dict[str, Any], ttl: Optional[int] = None):
        self._memory[conversation_id] = memory

    async def delete(self, conversation_id: str):
        if conversation_id in self._memory:
            del self._memory[conversation_id]

class RedisMemoryStore(BaseMemoryStore):
    """Redis-based storage (works for both standard Redis and Upstash)."""
    
    def __init__(self, tenant_id: str, url: str, default_ttl: int = 3600):
        try:
            import redis.asyncio as redis
            self.client = redis.from_url(url, decode_responses=True)
            self.tenant_id = tenant_id
            self.default_ttl = default_ttl
        except ImportError:
            logger.error("redis-py not installed. Please install with 'pip install redis'")
            raise

    def _get_key(self, conversation_id: str) -> str:
        return f"symvion:{self.tenant_id}:{conversation_id}"

    async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        key = self._get_key(conversation_id)
        data = await self.client.get(key)
        if not data:
            return None
            
        memory = json.loads(data)
        
        # Deserialize LangChain messages if present
        if "history" in memory:
            memory["history"] = messages_from_dict(memory["history"])
            
        return memory

    async def set(self, conversation_id: str, memory: Dict[str, Any], ttl: Optional[int] = None):
        key = self._get_key(conversation_id)
        
        # Deep copy to avoid mutating original
        serialized_memory = memory.copy()
        
        # Serialize LangChain messages
        if "history" in serialized_memory:
            serialized_memory["history"] = messages_to_dict(serialized_memory["history"])
            
        await self.client.set(
            key, 
            json.dumps(serialized_memory), 
            ex=ttl if ttl is not None else self.default_ttl
        )

    async def delete(self, conversation_id: str):
        key = self._get_key(conversation_id)
        await self.client.delete(key)

def get_memory_store(tenant_id: str, config: MemoryConfig) -> BaseMemoryStore:
    """Factory function to get the configured memory store."""
    if config.type == MemoryStoreType.REDIS or config.type == MemoryStoreType.UPSTASH:
        if not config.url:
            logger.warning(f"URL missing for {config.type} memory store. Falling back to local.")
            return InProcessMemoryStore()
        try:
            return RedisMemoryStore(tenant_id, config.url, config.ttl)
        except Exception as e:
            logger.error(f"Failed to initialize {config.type} memory store: {str(e)}. Falling back to local.")
            return InProcessMemoryStore()
    
    return InProcessMemoryStore()
