import asyncio
import os
import sys

os.environ["OPENAI_API_KEY"] = "sk-placeholder"

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from symvion.core.runtime import Symvion
from symvion.config.models import TenantConfig, MemoryConfig, MemoryStoreType, LoggingConfig

async def test_file_logging():
    log_file = "/tmp/symvion_test.log"
    if os.path.exists(log_file):
        os.remove(log_file)
        
    print(f"🚀 Testing file-based logging to {log_file}...")
    
    config = TenantConfig(
        tenant_id="log-test-tenant",
        llm_provider="mock",
        logging=LoggingConfig(file_path=log_file, show_json=True)
    )
    
    runtime = Symvion(config)
    
    await runtime.chat(
        tenant="log-test-tenant",
        message="Hello",
        session_id="log-session"
    )
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
            print("\nLog File Content:")
            print(content)
            if '"event": "REQUEST_START"' in content:
                print("\n✅ File logging verified!")
            else:
                print("\n❌ Log entry not found in file.")
    else:
        print(f"\n❌ Log file {log_file} was not created.")

if __name__ == "__main__":
    asyncio.run(test_file_logging())
