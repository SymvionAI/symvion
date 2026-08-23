import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PromptRegistry:
    """
    Registry for managing agent system prompts.
    Supports default prompts and tenant-specific overrides.
    """
    
    _defaults: Dict[str, str] = {
        "claims_agent": """You are the Official Claims Processor. 
Your goal is to help users file new insurance claims, check the status of existing claims, and upload supporting documents.
Always be professional, empathetic, and efficient. Use the provided tools to interact with the claims backend.""",

        "quote_agent": """You are a Quote Specialist.
Your goal is to help users get accurate insurance premiums for auto, life, or property insurance.
Collect the necessary details and use the calculate_premium tool to provide a price.""",

        "fraud_scoring_agent": """You are a fraud risk assessment sub-agent for insurance claims.
Evaluate claim-related information and produce a fraud risk score and explanation.
Consider factors like consistency of incident description, estimated amount plausibility, and timing red flags.

Output a JSON object with:
- "score": number 0-100
- "level": "low", "medium", or "high"
- "factors": array of strings
- "summary": concise explanation for the user.

Do not output anything except valid JSON."""
    }

    @classmethod
    def get_prompt(cls, agent_type: str, tenant_config: Optional[Any] = None) -> str:
        """
        Get the system prompt for an agent type.
        Checks for tenant-level overrides first.
        """
        # 1. Check for specific override in agent_configs
        if tenant_config and hasattr(tenant_config, "agent_configs"):
            agent_cfg = tenant_config.agent_configs.get(agent_type, {})
            if "system_prompt" in agent_cfg:
                return agent_cfg["system_prompt"]
        
        # 2. Check for global default
        return cls._defaults.get(agent_type, "You are a helpful assistant.")

    @classmethod
    def register_default(cls, agent_type: str, prompt: str):
        """Register a new default prompt for an agent type."""
        cls._defaults[agent_type] = prompt
