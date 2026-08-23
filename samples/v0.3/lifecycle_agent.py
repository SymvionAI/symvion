from typing import Dict, Any
from symvion.agents.base import BaseAgent
from symvion.core.context import TenantContext
from symvion.core.logger import logger

class SecureAuditAgent(BaseAgent):
    """
    Sample agent demonstrating v0.3 lifecycle hooks for audit logging.
    """
    
    async def before_execute(self, context: TenantContext, input_data: Dict[str, Any]):
        # Custom logic: Log that an audit is starting for a specific user ID in metadata
        user_id = context.metadata.get("user_id", "anonymous")
        logger.info("AUDIT_START", context, user_id=user_id)
        
        # Example: Enforce a specific metadata field
        if "session_token" not in context.metadata:
            logger.warn("AUDIT_SECURITY_WARNING", context, reason="Missing session_token")

    async def execute(self, context: TenantContext, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent_response": "Audit complete. No issues found.",
            "token_usage": {"total_tokens": 150}
        }

    async def after_execute(self, context: TenantContext, result: Dict[str, Any]):
        # Custom logic: Record a security event after execution
        logger.info("AUDIT_RECORDED", context, status="success")

    async def on_error(self, context: TenantContext, error: Exception):
        logger.error("AUDIT_ERROR", context, error=str(error))
