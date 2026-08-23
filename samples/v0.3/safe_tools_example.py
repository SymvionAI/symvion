import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from typing import Dict, Any
from symvion.tools.base import ToolSafetyWrapper
from symvion.core.context import TenantContext

# Dummy Tool 1: Unstable sync tool
def fetch_external_data(id: int):
    import time
    time.sleep(1) # Simulate slow sync call
    if id == 999:
        raise ValueError("Data not found")
    return {"id": id, "data": "Value X"}

# Dummy Tool 2: Async tool
async def process_async(payload: str):
    await asyncio.sleep(0.5)
    return f"Processed: {payload}"

async def run_safety_demo():
    ctx = TenantContext(tenant_id="demo-tenant")
    
    print("🚀 Invoking UNSTABLE SYNC tool via Safety Wrapper...")
    try:
        # ToolSafetyWrapper handles the sync-to-async thread pool execution
        result = await ToolSafetyWrapper.invoke(
            fetch_external_data,
            ctx,
            "fetch_data",
            {"id": 123},
            timeout=2.0
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Caught expected safety failure: {e}")

    print("\n🚀 Invoking ASYNC tool via Safety Wrapper...")
    async_res = await ToolSafetyWrapper.invoke(
        process_async,
        ctx,
        "process_async",
        {"payload": "Safety First"}
    )
    print(f"Result: {async_res}")

if __name__ == "__main__":
    asyncio.run(run_safety_demo())
