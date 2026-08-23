from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum
import yaml

class MemoryStoreType(str, Enum):
    LOCAL = "local"
    REDIS = "redis"
    UPSTASH = "upstash"

class MemoryConfig(BaseModel):
    type: MemoryStoreType = MemoryStoreType.LOCAL
    url: Optional[str] = None
    ttl: Optional[int] = 3600

class AgentRegistration(BaseModel):
    name: str
    description: str
    system_prompt: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=list)
    kb_ids: List[str] = Field(default_factory=list)
    execution_retries: int = 1
    execution_timeout: float = 300.0

class LoggingConfig(BaseModel):
    enabled: bool = True
    level: str = "INFO"
    file_path: Optional[str] = None
    show_json: bool = True

class PolicyRule(BaseModel):
    """A specific governance rule."""
    type: Literal["blacklist", "time_window", "custom_regex"]
    action: Literal["mask", "block", "log"]
    keywords: List[str] = Field(default_factory=list)
    pattern: Optional[str] = None
    start: Optional[str] = None # e.g., "08:00"
    end: Optional[str] = None   # e.g., "20:00"

class GovernancePolicy(BaseModel):
    """A collection of rules belonging to a specific company policy."""
    name: str
    rules: List[PolicyRule]

class GovernanceConfig(BaseModel):
    """Dynamic governance settings per tenant."""
    pii_patterns: Dict[str, str] = Field(default_factory=dict)
    policies: List[GovernancePolicy] = Field(default_factory=list)

class TenantConfig(BaseModel):
    tenant_id: str
    env_vars: Dict[str, str] = Field(default_factory=dict)
    summary_threshold: int = 20
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    llm_provider: str = "openai"
    llm_config: Dict[str, Any] = Field(default_factory=lambda: {"model": "gpt-4", "temperature": 0.7})
    streaming: bool = False
    router_type: str = "keyword"
    system_prompt: Optional[str] = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    enabled_agents: List[str] = Field(default_factory=list)
    enabled_middleware: List[str] = Field(default_factory=list)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    agent_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    iam_policies: Dict[str, List[str]] = Field(default_factory=lambda: {"tenant_admin": ["*"]})
    enable_hallucination_check: bool = False
    
    # Platform Interaction
    auth_token: Optional[str] = None
    backend_url: Optional[str] = None
    fraud_scoring: bool = False
    
    # Reasoning & Execution Guardrails
    max_tokens_per_turn: int = Field(default=4000)
    recursion_limit: int = Field(default=20)
    enable_thought_auditing: bool = False

    # HITL / controlled GenUI: only these tool names pause (empty = off).
    # Client/UI-marked tools always interrupt even if not listed.
    interrupt_before_tools: List[str] = Field(default_factory=list)
    # Allow HITL ``edit`` on server-side tools (not only mark_client_tool tools).
    # Default False: GenUI client tools can still edit; backend tools approve/reject only.
    trust_hitl_edit: bool = False

    # Security: do not trust client-supplied role / agent routing by default.
    # Hosts that already authenticate callers may set these True after mapping
    # verified JWT/gateway claims into metadata.
    trust_client_role: bool = False
    trust_client_agent_id: bool = False
    default_user_role: str = "user"
    # Optional HTTPS host allowlist for backend/MCP calls (empty = scheme + private-IP checks only).
    allowed_outbound_hosts: List[str] = Field(default_factory=list)
    # Directory under which MCP/local file_path reads are allowed.
    file_sandbox_dir: Optional[str] = None
    
    # Financial & Usage Governance
    token_limit: int = Field(default=2**31 - 1)  # High default limit
    current_usage: int = Field(default=0)
    
    # Internal context (not persisted in YAML usually, but tracked)
    config_path: Optional[str] = Field(default=None, exclude=True)


    @classmethod
    def from_yaml(cls, path: str) -> "TenantConfig":
        """Load tenant configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        config.config_path = path
        return config

    def save(self):
        """Save the current configuration back to the YAML file."""
        if not self.config_path:
            return
            
        # Convert to dict and ensure enums/etc are serialized to strings
        import json
        clean_dict = json.loads(self.model_dump_json(exclude_none=True))
            
        with open(self.config_path, 'w') as f:
            yaml.dump(clean_dict, f, sort_keys=False, default_flow_style=False)
