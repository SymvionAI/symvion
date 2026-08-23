"""
LegacyAgent: deep strategic doctrine evaluation using an LLM.
Used only when DoctrineRouter returns LEGACY_REASONING.
"""

import logging
import os
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.runtime.agents.legacy_agent.doctrine_models import (
    DoctrineEvaluationResult,
    DOCTRINE_FILTERS,
)
from app.runtime.agents.legacy_agent.doctrine_cache import get_principles
from app.runtime.agents.legacy_agent.doctrine_engine import _normalize_document_view

logger = logging.getLogger(__name__)

# JSON schema-friendly structure for LLM output (no nested Pydantic in filter_details for simplicity)
STRUCTURED_OUTPUT_INSTRUCTIONS = """
Respond with a single JSON object (no markdown) with exactly these keys:
- institutional_alignment_score: number 0-100 (overall alignment)
- scores: object with keys institutional_strength, governance_integrity, execution_readiness, strategic_leverage, talent_impact, capital_logic, ecosystem_relevance, vision_alignment; each value number 0-100
- governance_questions: array of strings (or empty array)
- execution_concerns: array of strings (or empty array)
- recommendation: string (brief doctrine-aligned recommendation)
"""


class LegacyAgent:
    """Performs deep strategic doctrine evaluation when routed by DoctrineRouter."""

    def __init__(self, tenant_id: str, config: Dict[str, Any] | None = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        model_name = self.config.get("model", os.getenv("LLM_MODEL", "gpt-4"))
        temperature = float(self.config.get("temperature", os.getenv("LLM_TEMPERATURE", "0.3")))
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    async def evaluate(
        self,
        agent_response: str,
        context: Dict[str, Any],
    ) -> DoctrineEvaluationResult:
        """
        Run strategic doctrine evaluation using the LLM.
        Input: agent output and context (document_analysis, optional document_extract).

        Returns:
            DoctrineEvaluationResult with scores, governance_questions, execution_concerns, recommendation.
        """
        import json

        principles = get_principles()
        document_text = _normalize_document_view(context, agent_response or "")

        principles_blob = "\n".join(
            f"- {p['name']}: " + "; ".join(p.get("rules") or [])
            for p in principles
        )
        filters_blob = ", ".join(DOCTRINE_FILTERS)

        system_prompt = (
            "You are a doctrine alignment evaluator. You assess whether decisions and content "
            "align with the founder's institutional doctrine. "
            "Evaluate the given content against these principle dimensions: " + filters_blob + ". "
            "Score each dimension 0-100 and provide a brief recommendation. "
            + STRUCTURED_OUTPUT_INSTRUCTIONS
        )

        user_content = (
            "Founder doctrine principles:\n" + principles_blob + "\n\n"
            "Content to evaluate:\n" + (document_text[:12000] or "(none)")
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            # Strip markdown code block if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
        except Exception as e:
            logger.exception("LegacyAgent LLM parse failed: %s", e)
            from app.runtime.agents.legacy_agent.doctrine_models import empty_evaluation
            return empty_evaluation()

        # Normalize to DoctrineEvaluationResult
        scores = data.get("scores") or {}
        for f in DOCTRINE_FILTERS:
            if f not in scores and isinstance(scores.get(f), (int, float)):
                continue
            if f not in scores:
                scores[f] = 0
            else:
                scores[f] = max(0, min(100, int(scores[f])))

        return DoctrineEvaluationResult(
            institutional_alignment_score=max(
                0, min(100, int(data.get("institutional_alignment_score", 0)))
            ),
            scores=scores,
            governance_questions=list(data.get("governance_questions") or []),
            execution_concerns=list(data.get("execution_concerns") or []),
            recommendation=str(data.get("recommendation") or ""),
            filter_details=None,
        )
