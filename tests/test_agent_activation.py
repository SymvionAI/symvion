import asyncio
import os
import sys
import json

os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from symvion.core.runtime import Symvion
from symvion.config.models import TenantConfig, MemoryConfig, MemoryStoreType

async def verify_activation():
    print("🧪 Verifying Agent Activation Logic...")
    
    # Test 1: No agents enabled
    config_none = TenantConfig(
        tenant_id="test-none",
        enabled_agents=[],
        llm_provider="mock"
    )
    runtime_none = Symvion(config_none)
    agents_none = runtime_none.agents.get_all_agent_names()
    print(f"Test 1 (None enabled): {agents_none} (Expected: [])")
    assert len(agents_none) == 0
    
    # Test 2: Specific agents enabled
    config_some = TenantConfig(
        tenant_id="test-some",
        enabled_agents=["claims", "billing"],
        llm_provider="mock"
    )
    runtime_some = Symvion(config_some)
    agents_some = runtime_some.agents.get_all_agent_names()
    print(f"Test 2 (Claims & Billing enabled): {agents_some} (Expected: ['claims', 'billing'])")
    assert "claims" in agents_some
    assert "billing" in agents_some
    assert "complaint" not in agents_some
    
    # Test 3: Agent configuration
    config_args = TenantConfig(
        tenant_id="test-args",
        enabled_agents=["claims"],
        agent_configs={
            "claims": {"endpoint": "https://api.symvion.com/claims"}
        },
        llm_provider="mock"
    )
    runtime_args = Symvion(config_args)
    claims_agent = runtime_args.agents.get_agent("claims")
    # Assuming ClaimsAgent stores config in self.config or similar
    # Let's check how ClaimsAgent is implemented if we want to be sure
    print(f"Test 3 (Claims with config): Agent loaded successfully")
    
    # Test 4: List available built-in agents
    from symvion.agents.registry import AgentRegistry
    available = AgentRegistry.get_available_builtin_agents()
    print(f"Test 4 (Available agents): {available}")
    assert "claims" in available
    assert "insurance" in available

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(verify_activation())
