import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class QuoteTools:
    """Tools for insurance quote generation."""

    def __init__(self, tenant_id: str, backend_url: Optional[str] = None, auth_token: Optional[str] = None):
        self.tenant_id = tenant_id
        self.backend_url = backend_url or "https://api.symvion.ai/quotes"
        self.auth_token = auth_token

    async def calculate_premium(self, coverage_type: str, coverage_amount: float, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the insurance premium for a given coverage.
        
        Args:
            coverage_type: Type of insurance (auto, life, property, etc.)
            coverage_amount: Desired coverage limit.
            details: Additional risk factors (age, location, history).
        """
        logger.info(f"Calculating premium for {coverage_type} at {self.backend_url}")
        
        # Mock calculation logic
        base_rate = {"auto": 0.05, "life": 0.02, "property": 0.03}.get(coverage_type.lower(), 0.04)
        annual_premium = coverage_amount * base_rate
        
        return {
            "annual_premium": annual_premium,
            "monthly_premium": annual_premium / 12,
            "currency": "NGN",
            "coverage_type": coverage_type,
            "tenant_id": self.tenant_id
        }

    async def save_quote(self, customer_email: str, quote_data: Dict[str, Any]) -> str:
        """Save a generated quote and return a reference ID."""
        import uuid
        quote_id = f"QT-{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"Saving quote {quote_id} for {customer_email}")
        return f"Quote saved successfully with ID: {quote_id}"

def get_quote_tools(tenant_id: str, backend_url: Optional[str] = None, auth_token: Optional[str] = None) -> QuoteTools:
    return QuoteTools(tenant_id, backend_url, auth_token)
