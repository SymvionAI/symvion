import asyncio
import os
import sys
import json

os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from symvion.core.runtime import Symvion
from symvion.config.models import TenantConfig, MemoryConfig, MemoryStoreType
from symvion.agents.weather_agent import WeatherAgent
from symvion.tools.weather_tools import get_weather

async def run_verification():
    print("🚀 Starting Symvion v0.3 Verification...")
    
    # 1. Setup Configuration
    config = TenantConfig(
        tenant_id="test-tenant-001",
        summary_threshold=5,
        llm_provider="mock", # Use mock for verification to avoid API costs/keys
        memory=MemoryConfig(type=MemoryStoreType.LOCAL)
    )
    
    # 2. Initialize Runtime
    runtime = Symvion(config)
    
    # 3. Register Tool
    runtime.tools.register("get_weather", get_weather)
    
    # 3. Register a v0.3 Code-based Agent
    weather_agent = WeatherAgent("test-tenant-001", {"name": "weather", "execution_retries": 2})
    runtime.agents._agents["weather"] = weather_agent
    
    # 4. Register a Backward-Compatible Dict-based Agent
    runtime.register_agent({
        "name": "billing",
        "description": "Handles billing inquiries",
        "system_prompt": "You are a billing assistant.",
        "input_schema": {},
        "output_schema": {}
    })
    
    print("\n--- Test 1: Code-based Agent with Tool Safety & Lifecycle ---")
    res1 = await runtime.chat(
        tenant="test-tenant-001",
        message="What is the weather in Lagos?",
        session_id="session-123"
    )
    print(f"Response: {res1['data']}")
    print(f"Request ID: {res1['request_id']}")
    
    print("\n--- Test 2: Routing & Multi-Tenant Context ---")
    res2 = await runtime.chat(
        tenant="test-tenant-001",
        message="Tell me about my bill",
        session_id="session-123"
    )
    print(f"Response: {res2['data']}")
    print(f"Agent: {res2['agent']}")

    print("\n--- Test 3: Structured Logs Requirement ---")
    print("Note: Check the terminal output above for JSON formatted logs from REQUEST_START, ROUTING_DECISION, TOOL_CALLED, etc.")

if __name__ == "__main__":
    asyncio.run(run_verification())
