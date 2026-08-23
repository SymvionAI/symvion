from abc import ABC, abstractmethod
from typing import List, Optional
from symvion.core.context import TenantContext

class BaseRouter(ABC):
    """
    Interface for agent routers.
    Allows for keyword, semantic, or LLM-based agent selection.
    """
    @abstractmethod
    async def route(self, context: TenantContext, message: str, available_agents: List[str]) -> str:
        """Return the name of the agent to route to."""
        pass
