"""
Agent modules for different business domains.
Each agent handles specific workflows (claims, registration, billing, etc.).
"""

from app.runtime.agents.registration_agent import RegistrationAgent
from app.runtime.agents.claims_agent import ClaimsAgent
from app.runtime.agents.billing_agent import BillingAgent
from app.runtime.agents.complaint_agent import ComplaintAgent
from app.runtime.agents.insurance_agent import InsuranceAgent
from app.runtime.agents.quote_generator_agent import QuoteGeneratorAgent
from app.runtime.agents.document_intelligence_agent import DocumentIntelligenceAgent
from app.runtime.agents.fraud_scoring_agent import FraudScoringAgent
from app.runtime.agents.legacy_agent import (
    LegacyAgent,
    doctrine_router_route,
    doctrine_engine_evaluate,
    get_principles,
    DoctrineEvaluationResult,
)

__all__ = [
    "RegistrationAgent",
    "ClaimsAgent",
    "BillingAgent",
    "ComplaintAgent",
    "InsuranceAgent",
    "QuoteGeneratorAgent",
    "DocumentIntelligenceAgent",
    "FraudScoringAgent",
    "LegacyAgent",
    "doctrine_router_route",
    "doctrine_engine_evaluate",
    "get_principles",
    "DoctrineEvaluationResult",
]
