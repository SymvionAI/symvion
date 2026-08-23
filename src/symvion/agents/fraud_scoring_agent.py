"""
Fraud scoring sub-agent.
Evaluates claim-related context for fraud risk and returns a structured score.
Designed to be invoked by the claims agent (e.g. after submit_claim or when user asks about fraud risk).
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from symvion.providers.factory import ProviderFactory
from symvion.prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)


class FraudScoringAgent:
    """Sub-agent that scores fraud risk from claim context."""

    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fraud scoring sub-agent.

        Args:
            tenant_id: Unique tenant identifier
            config: Optional agent-specific configuration (model, temperature)
        """
        self.tenant_id = tenant_id
        self.config = config or {}

        # Initialize LLM using ProviderFactory (inherits model/provider from config)
        self.llm = ProviderFactory.get_provider(
            self.config.get("provider", "openai"),
            self.config
        )

        self._system_prompt = PromptRegistry.get_prompt("fraud_scoring_agent", self.config)

    async def score(
        self,
        incident_description: Optional[str] = None,
        estimated_amount: Optional[float] = None,
        claim_type: Optional[str] = None,
        policy_number: Optional[str] = None,
        incident_date: Optional[str] = None,
        user_message: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compute a fraud risk score from claim context.

        Args:
            incident_description: Description of the incident
            estimated_amount: Claim amount or estimate
            claim_type: Type of claim (e.g. auto, property)
            policy_number: Policy identifier (optional)
            incident_date: Date of incident (optional)
            user_message: Raw user message for extra context (optional)
            **kwargs: Additional context (e.g. contact_info, claim_id)

        Returns:
            Dict with keys: score (0-100), level (low|medium|high), factors (list), summary (str).
            On error, returns a safe default with score 0 and summary describing the error.
        """
        # Build context string for the LLM
        parts = []
        if incident_description:
            parts.append(f"Incident description: {incident_description}")
        if estimated_amount is not None:
            parts.append(f"Estimated amount: {estimated_amount}")
        if claim_type:
            parts.append(f"Claim type: {claim_type}")
        if policy_number:
            parts.append(f"Policy number: {policy_number}")
        if incident_date:
            parts.append(f"Incident date: {incident_date}")
        if user_message:
            parts.append(f"User message (context): {user_message}")
        for k, v in kwargs.items():
            if v is not None and v != "":
                parts.append(f"{k}: {v}")

        if not parts:
            return {
                "score": 0,
                "level": "low",
                "factors": [],
                "summary": "No claim details provided; fraud risk cannot be assessed.",
            }

        context_text = "\n".join(parts)

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=f"Assess fraud risk for this claim context:\n\n{context_text}"
            ),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            text = (response.content or "").strip()

            # Try to extract JSON (handle markdown code blocks)
            if "```" in text:
                start = text.find("```")
                if "json" in text[: start + 10]:
                    start = text.find("\n", start) + 1
                end = text.find("```", start)
                text = text[start:end] if end > start else text
            parsed = json.loads(text)

            score = parsed.get("score", 0)
            level = (parsed.get("level") or "low").lower()
            if level not in ("low", "medium", "high"):
                level = "low" if score < 34 else "high" if score > 66 else "medium"
            factors = parsed.get("factors")
            if not isinstance(factors, list):
                factors = []
            summary = parsed.get("summary") or "No summary provided."

            return {
                "score": max(0, min(100, int(score))),
                "level": level,
                "factors": factors,
                "summary": summary,
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Fraud scoring JSON parse failed: %s", e)
            return {
                "score": 0,
                "level": "low",
                "factors": [],
                "summary": "Fraud assessment could not be completed; no risk factors were evaluated.",
            }
        except Exception as e:
            logger.error("Fraud scoring error: %s", e, exc_info=True)
            return {
                "score": 0,
                "level": "low",
                "factors": [],
                "summary": "An error occurred during fraud risk assessment.",
            }
