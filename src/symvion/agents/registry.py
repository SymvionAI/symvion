from typing import Dict, Any, Union, List, Optional
import logging
from symvion.config.models import AgentRegistration
from symvion.agents.base import BaseAgent

logger = logging.getLogger(__name__)

class AgentRegistry:
    BUILTIN_AGENTS = ["claims", "billing", "complaint", "insurance", "retrieval", "quote_generator"]

    def __init__(
        self,
        tenant_id: str = "default",
        llm_provider: str = "openai",
        llm_config: dict = None,
        streaming: bool = True,
        enabled_agents: List[str] = None,
        agent_configs: Dict[str, Dict[str, Any]] = None,
        allowed_outbound_hosts: Optional[List[str]] = None,
        file_sandbox_dir: Optional[str] = None,
        backend_url: Optional[str] = None,
        auth_token: Optional[str] = None,
    ):
        self._agents: Dict[str, Union[AgentRegistration, Any, BaseAgent]] = {}
        self.tenant_id = tenant_id
        self.llm_provider = llm_provider
        self.llm_config = llm_config or {}
        self.streaming = streaming
        self.enabled_agents = enabled_agents or []
        self.agent_configs = agent_configs or {}
        self.allowed_outbound_hosts = allowed_outbound_hosts or []
        self.file_sandbox_dir = file_sandbox_dir
        self.backend_url = backend_url
        self.auth_token = auth_token
        self._preload_builtin_agents()

    def _get_merged_config(self, agent_name: str) -> Dict[str, Any]:
        """Merge base LLM config with agent-specific overrides."""
        # Start with agent-specific config
        config = self.agent_configs.get(agent_name, {}).copy()
        
        # Inject base provider and model if not overridden
        if "provider" not in config:
            config["provider"] = self.llm_provider
            
        # Inherit model and temperature from base llm_config if not explicitly set for agent
        for key in ["model", "temperature", "max_tokens_per_turn"]:
            if key not in config and key in self.llm_config:
                config[key] = self.llm_config[key]
                
        # Inherit streaming flag
        if "streaming" not in config:
            config["streaming"] = self.streaming

        if "allowed_outbound_hosts" not in config:
            config["allowed_outbound_hosts"] = list(self.allowed_outbound_hosts)
        if "file_sandbox_dir" not in config and self.file_sandbox_dir:
            config["file_sandbox_dir"] = self.file_sandbox_dir
        if "backend_url" not in config and self.backend_url:
            config["backend_url"] = self.backend_url
        if "auth_token" not in config and self.auth_token:
            config["auth_token"] = self.auth_token
                
        return config

    def merge_agent_config(
        self, agent_name: str, extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge tenant security/LLM defaults with optional registration payload."""
        config = self._get_merged_config(agent_name)
        if extra:
            config.update(extra)
        return config
        
    def _preload_builtin_agents(self):
        # 1. Claims Agent
        if "claims" in self.enabled_agents:
            try:
                from symvion.agents.claims_agent import ClaimsAgent
                config = self._get_merged_config("claims")
                self._agents["claims"] = ClaimsAgent(self.tenant_id, config)
            except ImportError as e:
                logger.debug(f"Failed to preload claims agent: {e}")
            
        # 2. Billing Agent
        if "billing" in self.enabled_agents:
            try:
                from symvion.agents.billing_agent import BillingAgent
                config = self._get_merged_config("billing")
                self._agents["billing"] = BillingAgent(self.tenant_id, config)
            except ImportError:
                pass

        # 3. Complaint Agent
        if "complaint" in self.enabled_agents:
            try:
                from symvion.agents.complaint_agent import ComplaintAgent
                config = self._get_merged_config("complaint")
                self._agents["complaint"] = ComplaintAgent(self.tenant_id, config)
            except ImportError:
                pass
            
        # 4. Insurance Agent (Legacy)
        if "insurance" in self.enabled_agents:
            try:
                from symvion.agents.insurance_agent import InsuranceAgent
                config = self._get_merged_config("insurance")
                self._agents["insurance"] = InsuranceAgent(self.tenant_id, config)
            except ImportError:
                pass

        # 5. Retrieval Agent
        if "retrieval" in self.enabled_agents:
            try:
                from symvion.agents.retrieval_agent import RetrievalAgent
                config = self._get_merged_config("retrieval")
                self._agents["retrieval"] = RetrievalAgent(self.tenant_id, config)
            except ImportError as e:
                logger.debug(f"Failed to preload retrieval agent: {e}")

        # 6. Quote Generator Agent
        if "quote_generator" in self.enabled_agents:
            try:
                from symvion.agents.quote_agent import QuoteAgent
                config = self._get_merged_config("quote_generator")
                self._agents["quote_generator"] = QuoteAgent(self.tenant_id, config)
            except ImportError as e:
                logger.debug(f"Failed to preload quote agent: {e}")
            
    def register_via_api(self, registration: AgentRegistration):
        self._agents[registration.name] = registration

    def get_agent(self, name: str) -> Union[AgentRegistration, Any]:
        if name not in self._agents:
            raise ValueError(f"Agent {name} not found in registry.")
        return self._agents[name]
        
    def get_all_agent_names(self) -> List[str]:
        return list(self._agents.keys())
    
    @classmethod
    def get_available_builtin_agents(cls) -> List[str]:
        return cls.BUILTIN_AGENTS
