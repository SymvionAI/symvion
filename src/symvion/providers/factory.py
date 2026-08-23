from typing import Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
import os

class ProviderFactory:
    """
    Factory to initialize LangChain LLM providers.
    Ensures the system is LLM-agnostic.
    """
    
    @staticmethod
    def get_provider(provider_name: str, config: Dict[str, Any]) -> BaseChatModel:
        """
        Initialize and return a LangChain BaseChatModel.
        Uses langchain.chat_models.init_chat_model for universal provider support.
        """
        provider_name = provider_name.lower()
        
        if provider_name == "mock":
            from langchain_core.language_models.chat_models import SimpleChatModel
            from langchain_core.messages import BaseMessage
            from typing import List
            
            class MockChatModel(SimpleChatModel):
                def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> str:
                    return "Mock response"
                @property
                def _llm_type(self) -> str:
                    return "mock-chat"

            return MockChatModel()
            
        # Use LangChain's universal initializer
        # This supports: openai, anthropic, google_genai, fireworks, together, groq, etc.
        from langchain.chat_models import init_chat_model
        
        # Platform-specific keys that shouldn't be passed to the LLM provider
        PLATFORM_KEYS = {
            "model", "provider", "backend_url", "auth_token", "tenant_id", 
            "fraud_scoring", "system_prompt", "name", "description",
            "max_tokens_per_turn", "recursion_limit", "enable_thought_auditing",
            "allowed_outbound_hosts", "file_sandbox_dir", "agent_class",
            "tools", "kb_ids", "streaming", "trust_hitl_edit",
            "interrupt_before_tools", "iam_policies",
        }
        
        # Extract model name
        model_name = config.get("model", "gpt-4")
        if model_name == "claude-haiku-4.5":
            model_name = "claude-haiku-4-5-20251001"
            
        # Create init_config without platform-specific keys
        init_config = {k: v for k, v in config.items() if k not in PLATFORM_KEYS}
        
        # Ensure token usage is included in streams if relevant
        if provider_name == "openai":
             if "stream_options" not in init_config:
                 init_config["stream_options"] = {"include_usage": True}
        
        # Apply reasoning guardrails (Mapping max_tokens_per_turn to provider-specific keys)
        max_tokens_val = config.get("max_tokens_per_turn")
        if max_tokens_val:
             if "max_tokens" not in init_config and "max_completion_tokens" not in init_config:
                 if provider_name == "openai" and model_name.startswith("o1"):
                     init_config["max_completion_tokens"] = int(max_tokens_val)
                 else:
                     init_config["max_tokens"] = int(max_tokens_val)

        return init_chat_model(
            model=model_name,
            model_provider=provider_name,
            **init_config
        )

    @staticmethod
    def from_env(tenant_id: str) -> BaseChatModel:
        """
        Create a provider from environment variables for a specific tenant.
        """
        provider = os.getenv(f"SYM_PROVIDER_{tenant_id.upper()}", os.getenv("SYM_DEFAULT_PROVIDER", "openai"))
        model = os.getenv(f"SYM_MODEL_{tenant_id.upper()}", os.getenv("LLM_MODEL", "gpt-4"))
        temp = float(os.getenv(f"SYM_TEMP_{tenant_id.upper()}", os.getenv("LLM_TEMPERATURE", "0.7")))
        
        config = {
            "model": model,
            "temperature": temp
        }
        
        return ProviderFactory.get_provider(provider, config)
