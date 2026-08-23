from typing import Optional, Any, Dict

class SymvionError(Exception):
    """Base exception for all Symvion errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class AgentExecutionError(SymvionError):
    """Raised when an agent fails to execute."""
    pass

class ToolExecutionError(SymvionError):
    """Raised when a tool fails to execute."""
    pass

class RoutingError(SymvionError):
    """Raised when the router fails to select an agent."""
    pass

class TenantNotFoundError(SymvionError):
    """Raised when a requested tenant is not configured."""
    pass

class ConfigurationError(SymvionError):
    """Raised when there is an issue with the system configuration."""
    pass
