from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import time
from symvion.core.context import TenantContext
from symvion.core.logger import logger
from symvion.core.exceptions import AgentExecutionError
from langchain_core.runnables import RunnableConfig
from symvion.providers.factory import ProviderFactory

class BaseAgent(ABC):
    """
    Production-grade Base Agent with structured lifecycle hooks.
    Supports retries, timeouts, and multi-tenant context.
    """
    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        self.tenant_id = tenant_id
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.description = config.get("description", "Agent")
        
        # Sanitize config to avoid passing custom fields like 'kb_ids' to LLM constructors
        llm_config = {k: v for k, v in config.items() if k not in ["name", "description", "system_prompt", "tools", "kb_ids", "agent_class", "provider"]}
        
        # Initialize LLM using ProviderFactory (standard initialization for all v0.3 agents)
        self.llm = ProviderFactory.get_provider(
            config.get("provider", "openai"),
            llm_config
        )

    async def before_execute(self, context: TenantContext, input_data: Dict[str, Any]):
        """Lifecycle hook before execution."""
        logger.info("AGENT_BEFORE_EXECUTE", context, agent_name=self.name)

    @abstractmethod
    async def execute(self, context: TenantContext, input_data: Dict[str, Any], tools: Optional[Any] = None, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Internal execution logic to be implemented by subclasses."""
        pass

    async def after_execute(self, context: TenantContext, result: Dict[str, Any], duration: float):
        """Lifecycle hook after execution."""
        logger.info("AGENT_AFTER_EXECUTE", context, agent_name=self.name, duration=duration, status="success")

    async def on_error(self, context: TenantContext, error: Exception):
        """Lifecycle hook on error."""
        logger.error("AGENT_ERROR", context, agent_name=self.name, error=str(error))

    async def run(self, context: TenantContext, input_data: Dict[str, Any], tools: Optional[Any] = None, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Public entry point for agent execution.
        Orchestrates lifecycle, retries, and timeouts.
        """
        # Load from config or env overrides
        retries = int(self.config.get("execution_retries", 1))
        timeout = float(self.config.get("execution_timeout", 300.0))
        
        start_time = time.perf_counter()
        
        for attempt in range(retries):
            try:
                await self.before_execute(context, input_data)
                
                # Execute with timeout handler
                result = await asyncio.wait_for(
                    self.execute(context, input_data, tools, config=config),
                    timeout=timeout
                )
                
                # Ensure standard output format
                if "agent_response" not in result:
                    result["agent_response"] = result.get("content", "")
                
                duration = time.perf_counter() - start_time
                await self.after_execute(context, result, duration)
                return result
                
            except asyncio.TimeoutError:
                logger.warning("AGENT_TIMEOUT", context, agent_name=self.name, attempt=attempt+1)
                if attempt == retries - 1:
                    err = AgentExecutionError(f"Agent {self.name} timed out after {timeout}s")
                    await self.on_error(context, err)
                    raise err
            except Exception as e:
                logger.warning("AGENT_ATTEMPT_FAILED", context, agent_name=self.name, attempt=attempt+1, error=str(e))
                if attempt == retries - 1:
                    await self.on_error(context, e)
                    if isinstance(e, AgentExecutionError):
                        raise
                    raise AgentExecutionError(f"Agent {self.name} failed: {str(e)}") from e
                
                # Wait before retry (exponential backoff)
                sleep_time = (2 ** attempt) * 0.5
                await asyncio.sleep(sleep_time)
        
        # Should not be reachable but for completeness
        raise AgentExecutionError(f"Agent {self.name} failed after all retries.")
