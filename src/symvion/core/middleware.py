import re
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from symvion.core.context import TenantContext
from symvion.core.exceptions import AgentExecutionError
from symvion.core.logger import logger

class BaseMiddleware(ABC):
    """Abstract base class for all Symvion middleware."""
    
    @abstractmethod
    async def preprocess(self, context: TenantContext, data: Dict[str, Any]) -> Tuple[TenantContext, Dict[str, Any]]:
        """Hook called before the request reaches the orchestration graph."""
        pass

    @abstractmethod
    async def postprocess(self, context: TenantContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called after the graph has returned a result."""
        pass

class PIIMasker(BaseMiddleware):
    """Detects and masks PII in user messages."""
    
    # Simple regex-based patterns for common PII
    DEFAULTS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b"
    }

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        self.patterns = {**self.DEFAULTS, **(custom_patterns or {})}

    async def preprocess(self, context: TenantContext, data: Dict[str, Any]) -> Tuple[TenantContext, Dict[str, Any]]:
        message = data.get("message", "")
        if not message:
            return context, data
            
        masked_message = message
        masked_count = 0
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, masked_message)
            if matches:
                masked_message = re.sub(pattern, f"[{pii_type}_REDACTED]", masked_message)
                masked_count += len(matches)
        
        if masked_count > 0:
            logger.info("PII_REDACTED", context, count=masked_count)
            data["message"] = masked_message
            
        return context, data

    async def postprocess(self, context: TenantContext, result: Dict[str, Any]) -> Dict[str, Any]:
        # Usually we don't mask the assistant response unless specifically required
        return result

class UsageValidator(BaseMiddleware):
    """Enforces per-tenant token usage caps."""
    
    async def preprocess(self, context: TenantContext, data: Dict[str, Any]) -> Tuple[TenantContext, Dict[str, Any]]:
        curr_usage = context.metadata.get("current_usage", 0)
        token_limit = context.metadata.get("token_limit", 2**31 - 1)
        
        if curr_usage >= token_limit:
            err_msg = f"Token Limit Exceeded: Tenant '{context.tenant_id}' has reached its lifecycle limit ({token_limit} tokens)"
            logger.error("TOKEN_LIMIT_EXCEEDED", context, current=curr_usage, limit=token_limit)
            raise AgentExecutionError(err_msg)
            
        return context, data

    async def postprocess(self, context: TenantContext, result: Dict[str, Any]) -> Dict[str, Any]:
        return result

class PolicyMiddleware(BaseMiddleware):
    """Evaluates custom business rules based on keyword blacklists, time windows, etc."""
    
    def __init__(self, policies: List[Any]):
        self.policies = policies

    async def preprocess(self, context: TenantContext, data: Dict[str, Any]) -> Tuple[TenantContext, Dict[str, Any]]:
        message = data.get("message", "")
        if not message:
            return context, data
            
        for policy in self.policies:
            for rule in policy.get("rules", []):
                rule_type = rule.get("type")
                action = rule.get("action")
                
                # 1. Blacklist Logic
                if rule_type == "blacklist":
                    for kw in rule.get("keywords", []):
                        if kw.lower() in message.lower():
                            if action == "block":
                                logger.warning("POLICY_BLOCK", context, policy=policy.get("name"), keyword=kw)
                                raise AgentExecutionError(f"Rejected by Policy: '{policy.get('name')}' (Sensitive keyword detected)")
                            elif action == "mask":
                                message = re.sub(re.escape(kw), "[REDACTED]", message, flags=re.IGNORECASE)
                                data["message"] = message

                # 2. Time Window Logic
                elif rule_type == "time_window":
                    now = datetime.now().strftime("%H:%M")
                    start = rule.get("start")
                    end = rule.get("end")
                    if start and end:
                        # Simple string comparison for time windows
                        if not (start <= now <= end):
                            if action == "block":
                                logger.warning("POLICY_TIME_BLOCK", context, now=now, window=f"{start}-{end}")
                                raise AgentExecutionError(f"Rejected by Policy: '{policy.get('name')}' (Current time {now} is outside allowed window {start}-{end})")
                            
        return context, data

    async def postprocess(self, context: TenantContext, result: Dict[str, Any]) -> Dict[str, Any]:
        return result

class HallucinationValidator(BaseMiddleware):
    """
    Uses a secondary LLM call to verify the groundedness of the AI response.
    Checks for contradictions or unverified claims against the context.
    """
    def __init__(self, llm_provider: str, llm_config: Dict[str, Any]):
        from symvion.providers.factory import ProviderFactory
        self.llm = ProviderFactory.get_provider(llm_provider, {**llm_config, "temperature": 0})

    async def preprocess(self, context: TenantContext, data: Dict[str, Any]) -> Tuple[TenantContext, Dict[str, Any]]:
        return context, data

    async def postprocess(self, context: TenantContext, result: Dict[str, Any]) -> Dict[str, Any]:
        response_text = result.get("data", "")
        # Get the original message from the context-metadata if we stored it there or pass it via result
        original_query = context.metadata.get("last_query", "Unknown Query")
        
        prompt = f"""You are a Fact-Checker. Verify if the following AI response is grounded and accurate.
        
        USER QUERY: {original_query}
        AI RESPONSE: {response_text}
        
        Provide a score from 0 to 1 where 1 is perfectly grounded and 0 is complete hallucination.
        Return ONLY a JSON object: {{"score": float, "reason": "concise reason"}}"""
        
        try:
            from langchain_core.messages import HumanMessage
            check_res = await self.llm.ainvoke([HumanMessage(content=prompt)])
            import json
            # Extract JSON from potential markdown markers
            clean_content = check_res.content.replace("```json", "").replace("```", "").strip()
            score_data = json.loads(clean_content)
            
            result["hallucination_check"] = score_data
            logger.info("HALLUCINATION_CHECK_COMPLETE", context, score=score_data.get("score"))
        except Exception as e:
            # Fail closed: treat verification failures as ungrounded.
            logger.error("HALLUCINATION_CHECK_FAILED", context, error=str(e))
            result["hallucination_check"] = {
                "score": 0.0,
                "reason": "Verification failed; treated as unverified",
                "verified": False,
            }
        
        return result

class MiddlewareChain:
    """Orchestrates the execution of multiple middleware components."""
    
    def __init__(self, middleware_list: List[BaseMiddleware]):
        self.middleware = middleware_list

    async def run_preprocess(self, context: TenantContext, data: Dict[str, Any]) -> Tuple[TenantContext, Dict[str, Any]]:
        for m in self.middleware:
            context, data = await m.preprocess(context, data)
        return context, data

    async def run_postprocess(self, context: TenantContext, result: Dict[str, Any]) -> Dict[str, Any]:
        for m in reversed(self.middleware):
            result = await m.postprocess(context, result)
        return result
