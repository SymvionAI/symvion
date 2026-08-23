import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from symvion.config.models import TenantConfig

class ConfigRegistry:
    """
    Utility for loading and managing multi-tenant configurations from central files.
    """
    @staticmethod
    def load_tenant_config(path: str, tenant_id: str) -> TenantConfig:
        """
        Loads a specific tenant's configuration from a registry-styled YAML file.
        
        Expected structure:
        tenants:
          tenant-id-1:
            llm_provider: ...
          tenant-id-2:
            ...
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at {path}")
            
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        if not data or "tenants" not in data:
            raise ValueError(f"Invalid config format in {path}. Missing 'tenants' key.")
            
        tenant_data = data["tenants"].get(tenant_id)
        if not tenant_data:
            raise KeyError(f"Tenant '{tenant_id}' not found in {path}")
            
        # Ensure tenant_id is set in the data
        tenant_data["tenant_id"] = tenant_id
        
        config = TenantConfig(**tenant_data)
        config.config_path = path
        return config
