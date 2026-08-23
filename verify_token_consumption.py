import os
import yaml
import asyncio
from symvion.config.models import TenantConfig
from symvion.core.runtime import Symvion

async def verify_tokens():
    print("--- Phase 4: Token Consumption Verification ---")
    
    # 1. Setup Mock Config
    config_path = "/tmp/tenant_tokens.yaml"
    with open(config_path, "w") as f:
        yaml.dump({
            "tenant_id": "token-test",
            "llm_provider": "openai",
            "llm_config": {"model": "gpt-4o-mini"},
            "token_limit": 500, # Very low limit for testing
            "current_usage": 0
        }, f)
        
    config = TenantConfig.from_yaml(config_path)
    runtime = Symvion(config)
    
    # 2. Run first request
    print("\nRunning first request...")
    res1 = await runtime.chat("token-test", "Hello! Just testing token tracking.")
    
    # Check usage
    usage1 = res1["tokens"].get("total_tokens", 0)
    print(f"Request 1 tokens used: {usage1}")
    
    config_reloaded = TenantConfig.from_yaml(config_path)
    print(f"Persisted usage: {config_reloaded.current_usage}")
    
    if config_reloaded.current_usage != usage1:
        print("❌ FAILED: Usage not persisted correctly")
        return

    # 3. Run second request (should still pass if under 500)
    print("\nRunning second request...")
    res2 = await runtime.chat("token-test", "Another small request.")
    usage2 = res2["tokens"].get("total_tokens", 0)
    print(f"Request 2 tokens used: {usage2}")
    print(f"Total usage now: {runtime.config.current_usage}")

    # 4. Trigger limit
    print("\nAttempting to exceed limit...")
    # Set limit very low manually to trigger it if not already hit
    runtime.config.token_limit = 10 
    runtime.config.save()
    
    try:
        await runtime.chat("token-test", "This should fail.")
        print("❌ FAILED: Limit was not enforced!")
    except Exception as e:
        print(f"✅ SUCCESS: Caught expected error: {str(e)}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    asyncio.run(verify_tokens())
