from fastapi import FastAPI, Depends, Header, HTTPException
from symvion.core.factory import SymvionFactory
from symvion.config.models import TenantConfig
from typing import Optional

app = FastAPI(title="Symvion Multi-Tenant Gateway")

# Simulated database/config loader
async def mock_db_config_loader(tenant_id: str) -> TenantConfig:
    # In a real app, you'd query a database or vault here
    if tenant_id == "unknown":
        raise ValueError("Tenant not found in DB")
        
    return TenantConfig(
        tenant_id=tenant_id,
        llm_provider="openai",
        llm_config={"model": "gpt-4"}
    )

async def get_runtime(x_tenant_id: str = Header(...)):
    """Dependency that injects the correct Symvion runtime for the tenant."""
    try:
        return await SymvionFactory.get_runtime(x_tenant_id, mock_db_config_loader)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/chat")
async def chat(
    message: str,
    runtime = Depends(get_runtime),
    session_id: Optional[str] = "default"
):
    # runtime is the cached Symvion instance for this specific tenant
    result = await runtime.chat(
        tenant=runtime.config.tenant_id,
        message=message,
        session_id=session_id
    )
    return result

@app.post("/reload/{tenant_id}")
async def reload_tenant(tenant_id: str):
    """Admin endpoint to force reload a tenant config."""
    await SymvionFactory.reload_runtime(tenant_id, mock_db_config_loader)
    return {"status": "reloaded", "tenant_id": tenant_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
