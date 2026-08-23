from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid

@dataclass(frozen=True)
class TenantContext:
    """
    Context object that carries tenant-specific information throughout the request lifecycle.
    Prevents scope leakage and enables rich observability.
    """
    tenant_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "metadata": self.metadata
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Helper to get values from metadata or attributes."""
        return self.metadata.get(key, getattr(self, key, default))
