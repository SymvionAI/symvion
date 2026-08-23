from typing import List, Optional, Any, Dict
from symvion.routers.base import BaseRouter
from symvion.core.context import TenantContext
from symvion.core.logger import logger
from langchain_core.runnables import RunnableConfig

class KeywordRouter(BaseRouter):
    """
    A simple keyword-based router.
    Improved with word-level matching and basic pluralization support.
    """
    async def route(self, context: TenantContext, message: str, available_agents: List[str]) -> str:
        logger.info("ROUTING_START", context, message=message[:50])
        
        message_clean = message.lower()
        words = set(message_clean.split())
        
        # 1. Exact or plural match
        for agent_name in available_agents:
            clean_name = agent_name.lower()
            if clean_name in words or f"{clean_name}s" in words or clean_name in message_clean:
                logger.info("ROUTING_DECISION", context, agent_selected=agent_name, strategy="keyword_smart")
                return {"agent": agent_name, "token_usage": {}}
        
        logger.info("ROUTING_DECISION", context, agent_selected="general", strategy="fallback")
        return {"agent": "general", "token_usage": {}}

class LLMRouter(BaseRouter):
    """
    An LLM-based intent classifier.
    Uses descriptions to route accurately even without keyword matches.
    """
    def __init__(self, llm_provider):
        self.llm = llm_provider

    async def route(self, context: TenantContext, message: str, agent_info: List[dict], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Routes based on agent descriptions.
        agent_info: List of dicts with 'name' and 'description'.
        """
        logger.info("LLM_ROUTING_START", context)
        
        choices = "\n".join([f"- {a['name']}: {a['description']}" for a in agent_info])
        
        prompt = f"""You are a precise intent classifier and router.
Given the user message and a list of available AI agents, determine which agent is best suited to handle the request.

AVAILABLE AGENTS:
{choices}
- general: Use this for greetings, general names, or if no other agent fits well.

USER MESSAGE: "{message}"

Return ONLY the name of the selected agent. Do not explain your choice."""

        from langchain_core.messages import HumanMessage, SystemMessage
        
        try:
            # Use astream and accumulate to allow outer graph events to capture router tokens
            response_text = ""
            usage = {}
            async for chunk in self.llm.astream([
                SystemMessage(content="You are a router. Output only the agent name."),
                HumanMessage(content=prompt)
            ], config=config):
                response_text += chunk.content
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = chunk.usage_metadata
            
            selected = response_text.strip().lower()
            
            # Validate selection
            valid_names = [a['name'].lower() for a in agent_info] + ["general"]
            if selected in valid_names:
                logger.info("LLM_ROUTING_DECISION", context, agent_selected=selected)
                # Return original case if it matches a valid name
                final_agent = "general"
                for a in agent_info:
                    if a['name'].lower() == selected:
                        final_agent = a['name']
                        break
                return {"agent": final_agent, "token_usage": usage}
            
            logger.warning("LLM_ROUTER_INVALID_SELECTION", context, response=selected)
            return {"agent": "general", "token_usage": usage}
        except Exception as e:
            logger.error("LLM_ROUTING_FAILED", context, error=str(e))
            return {"agent": "general", "token_usage": {}}
