from typing import Dict, Optional, Callable, Awaitable
import asyncio
from symvion.core.runtime import Symvion
from symvion.config.models import TenantConfig
from symvion.core.logger import logger

class SymvionFactory:
    """
    Singleton-style factory for managing multi-tenant Symvion runtimes.
    Caches instances to optimize memory and performance (Prevents re-compiling LangGraph for every request).
    """
    _instances: Dict[str, Symvion] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get_runtime(
        cls, 
        tenant_id: str, 
        config_loader: Optional[Callable[[str], Awaitable[TenantConfig]]] = None
    ) -> Symvion:
        """
        Get or create a Symvion runtime for a specific tenant.
        
        Args:
            tenant_id: The unique identifier for the tenant.
            config_loader: An async function that takes a tenant_id and returns a TenantConfig.
                          Required if the runtime is not already cached.
        """
        async with cls._lock:
            if tenant_id in cls._instances:
                return cls._instances[tenant_id]
            
            if not config_loader:
                logger.error("FACTORY_MISSING_LOADER", None, tenant_id=tenant_id)
                raise ValueError(f"Runtime for tenant '{tenant_id}' is not initialized and no config_loader was provided.")
                
            logger.info("FACTORY_INITIALIZING_RUNTIME", None, tenant_id=tenant_id)
            
            # Load config and initialize
            try:
                config = await config_loader(tenant_id)
                runtime = Symvion(config)
                cls._instances[tenant_id] = runtime
                return runtime
            except Exception as e:
                logger.error("FACTORY_INIT_FAILED", None, tenant_id=tenant_id, error=str(e))
                raise

    @classmethod
    async def reload_runtime(cls, tenant_id: str, config_loader: Callable[[str], Awaitable[TenantConfig]]):
        """Force a reload of a tenant's runtime (e.g., after a config change)."""
        async with cls._lock:
            cls._instances.pop(tenant_id, None)
            return await cls.get_runtime(tenant_id, config_loader)

    @classmethod
    def clear_all(cls):
        """Shutdown and clear all cached runtimes."""
        cls._instances.clear()
        logger.info("FACTORY_CLEARED", None)
