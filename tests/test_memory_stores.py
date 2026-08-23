import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from symvion.config.models import MemoryConfig, MemoryStoreType, TenantConfig

# Mock redis module for tests
mock_redis_mod = MagicMock()
sys.modules["redis"] = mock_redis_mod
sys.modules["redis.asyncio"] = mock_redis_mod

from symvion.memory.memory_store import get_memory_store, InProcessMemoryStore, RedisMemoryStore

async def test_local_store():
    print("Testing InProcessMemoryStore...")
    config = MemoryConfig(type=MemoryStoreType.LOCAL)
    store = get_memory_store("test-tenant", config)
    assert isinstance(store, InProcessMemoryStore)
    
    await store.set("conv1", {"history": ["hello"]})
    data = await store.get("conv1")
    assert data["history"] == ["hello"]
    
    await store.delete("conv1")
    data = await store.get("conv1")
    assert data is None
    print("InProcessMemoryStore passed!")

async def test_redis_factory():
    print("Testing RedisMemoryStore factory...")
    config = MemoryConfig(type=MemoryStoreType.REDIS, url="redis://localhost:6379")
    
    # Mock redis to avoid dependency issues during tests if not installed/running
    with patch("redis.asyncio.from_url") as mock_redis:
        store = get_memory_store("test-tenant", config)
        assert isinstance(store, RedisMemoryStore)
        assert store.tenant_id == "test-tenant"
        # We check the internal client or call a method
    print("RedisMemoryStore factory passed!")

async def test_runtime_integration():
    print("Testing Symvion runtime integration...")
    from symvion.core.runtime import Symvion
    
    config = TenantConfig(tenant_id="test-tenant")
    runtime = Symvion(config=config)
    
    assert isinstance(runtime.memory_store, InProcessMemoryStore)
    
    # Mock graph to avoid LLM calls
    runtime.graph = MagicMock()
    
    # Create the actual mock for ainvoke
    async def mock_ainvoke(state, config=None):
        return {"agent_response": "Hi", "current_agent": "default", "token_usage": {}}
    
    runtime.graph.ainvoke = mock_ainvoke
    
    await runtime.chat("test-tenant", message="Hello", session_id="session1")
    
    # Verify it was stored in memory
    data = await runtime.memory_store.get("session1")
    assert data is not None
    assert data["current_agent"] == "default"
    print("Symvion runtime integration passed!")

async def main():
    try:
        await test_local_store()
        await test_redis_factory()
        await test_runtime_integration()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
