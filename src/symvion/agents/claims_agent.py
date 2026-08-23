"""
Claims processing agent.
Handles insurance claims workflows and interactions.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from symvion.tools.claims_tools import get_claims_tools
from symvion.agents.fraud_scoring_agent import FraudScoringAgent
from symvion.providers.factory import ProviderFactory
from symvion.agents.base import BaseAgent
from symvion.utils.helpers import ensure_messages, filter_allowed_tools
from symvion.core.context import TenantContext
from symvion.prompts.registry import PromptRegistry
from langchain_core.runnables import RunnableConfig
from symvion.tools.base import ToolSafetyWrapper
from symvion.tools.hitl import reraise_if_hitl_pause, run_tool_with_hitl

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class ClaimsAgent(BaseAgent):
    """Agent for processing insurance claims."""

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        super().__init__(tenant_id, config)
        """
        Initialize claims agent.

        Args:
            tenant_id: Unique tenant identifier
            config: Agent-specific configuration
        """
        self.tenant_id = tenant_id
        self.config = config
        self.description = "Official claims processor. Handles filing new insurance claims, checking claim status, and uploading supporting documents."

        # Initialize claims tools
        backend_url = config.get("backend_url") or os.getenv("BACKEND_URL")
        auth_token = config.get("auth_token")  # Get auth token from config
        self.claims_tools = get_claims_tools(
            tenant_id,
            backend_url,
            auth_token,
            allowed_outbound_hosts=config.get("allowed_outbound_hosts"),
        )

        # Fraud scoring sub-agent (used via get_fraud_score tool)
        self.fraud_scoring_agent = FraudScoringAgent(
            tenant_id, config.get("fraud_scoring") or {}
        )

        # Create LangChain tools from claims tools (includes fraud scoring)
        self.tools = self._create_tools()

        self.tools = self._create_tools()
        # LLM is initialized in BaseAgent.__init__
        self.system_prompt = PromptRegistry.get_prompt("claims_agent", self.config)

    def _create_tools(self) -> List[StructuredTool]:
        """Create LangChain tools from claims tools."""
        return [
            StructuredTool.from_function(
                func=self.claims_tools.submit_claim,
                name="submit_claim",
                description="File a new claim. Need: policy_number, incident_date (YYYY-MM-DD), description, type.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.get_claim_status,
                name="get_claim_status",
                description="Check status of an existing claim.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.get_claim_details,
                name="get_claim_details",
                description="Get full details for a claim.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.update_claim,
                name="update_claim",
                description="Update claim info.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.list_claims,
                name="list_claims",
                description="List customer claims.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.upload_claim_document,
                name="upload_claim_document",
                description="Attach document to claim.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.calculate_claim_estimate,
                name="calculate_claim_estimate",
                description="Get claim estimate.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.get_claim_timeline,
                name="get_claim_timeline",
                description="View claim history.",
            ),
            StructuredTool.from_function(
                func=self._get_fraud_score,
                name="get_fraud_score",
                description="Check claim fraud risk.",
            ),
        ]

    def _build_system_prompt(self) -> str:
        """Construct the system prompt for the claims assistant."""
        return PromptRegistry.get_prompt("claims_agent", self.config)

    async def _get_fraud_score(
        self,
        incident_description: str = "",
        estimated_amount: Optional[float] = None,
        claim_type: str = "",
        policy_number: str = "",
        incident_date: str = "",
        user_message: str = "",
    ) -> Dict[str, Any]:
        """Invoke the fraud scoring sub-agent and return the assessment."""
        return await self.fraud_scoring_agent.score(
            incident_description=incident_description or None,
            estimated_amount=estimated_amount,
            claim_type=claim_type or None,
            policy_number=policy_number or None,
            incident_date=incident_date or None,
            user_message=user_message or None,
        )

    async def execute(self, context: TenantContext, input_data: Dict[str, Any], tools: Optional[Any] = None, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Process claims-related input with tool support.
        """
        try:
            message = input_data.get("message", "")
            history = input_data.get("history", [])

            # Use unified active governance to filter tools based on IAM policies
            allowed_tools = filter_allowed_tools(context, self.tools)
            llm_with_tools = self.llm.bind_tools(allowed_tools)

            # Build conversation history
            messages = [SystemMessage(content=self.system_prompt)]

            # Inject document analysis if present
            document_analysis = context.get("document_analysis")
            if document_analysis:
                messages.append(
                    SystemMessage(
                        content=(
                            "Source of truth from uploaded document:\n\n"
                            f"{document_analysis}\n\n"
                            "Extract details (policy_number, incident_date, etc.) from this summary to file claims."
                        )
                    )
                )

            # Add conversation history (handling both dicts and BaseMessage objects)
            for hist_msg in history[-10:]:
                if isinstance(hist_msg, dict):
                    role = hist_msg.get("role")
                    content = hist_msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                elif isinstance(hist_msg, HumanMessage):
                    messages.append(hist_msg)
                elif isinstance(hist_msg, AIMessage):
                    messages.append(hist_msg)
                elif isinstance(hist_msg, SystemMessage):
                    messages.append(hist_msg)

            # Add current user message
            messages.append(HumanMessage(content=message))

            # Get response from LLM using astream to enable token events
            response = None
            async for chunk in llm_with_tools.astream(messages, config=config):
                if response is None:
                    response = chunk
                else:
                    response += chunk

            # Check if LLM wants to call tools
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute tool calls
                tool_results = []
                iam = context.metadata.get("iam_policies")
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    call_id = tool_call.get("id", f"call_{tool_name}")

                    logger.info(f"Claims agent tool call: {tool_name}")

                    # Find the tool and execute it
                    matched_tool = None
                    tool_func = None
                    for tool in allowed_tools:
                        if tool.name == tool_name:
                            matched_tool = tool
                            tool_func = tool.func
                            break

                    if tool_func:
                        try:
                            _func = tool_func

                            async def _exec(_f=_func, _n=tool_name, **kwargs):
                                return await ToolSafetyWrapper.invoke(
                                    _f,
                                    context,
                                    _n,
                                    kwargs or {},
                                    iam_policies=iam,
                                )

                            result = await run_tool_with_hitl(
                                tool_name=tool_name,
                                tool_args=tool_args or {},
                                call_id=call_id,
                                execute=_exec,
                                config=config,
                                tool=matched_tool,
                            )
                            tool_results.append({"tool": tool_name, "result": result})
                        except Exception as tool_error:
                            reraise_if_hitl_pause(tool_error)
                            logger.error(
                                "Error in tool %s: %s",
                                tool_name,
                                type(tool_error).__name__,
                            )
                            tool_results.append(
                                {"tool": tool_name, "error": "tool execution failed"}
                            )
                    else:
                        tool_results.append({"tool": tool_name, "error": "Tool not found"})

                # Add tool results to messages
                messages.append(response)
                for i, tool_call in enumerate(response.tool_calls):
                    tool_result = tool_results[i] if i < len(tool_results) else {"error": "No result"}
                    tool_call_id = tool_call.get("id", f"call_{i}")

                    result_str = json.dumps(tool_result.get("result")) if "result" in tool_result else f"Error: {tool_result.get('error')}"
                    messages.append(ToolMessage(content=result_str, tool_call_id=tool_call_id))

                # Get final response after tool execution using astream
                final_response = None
                async for chunk in self.llm.astream(messages, config=config):
                    if final_response is None:
                        final_response = chunk
                    else:
                        final_response += chunk
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        usage = chunk.usage_metadata
                
                response_content = final_response.content
            else:
                response_content = response.content

            return {
                "content": response_content,
                "agent_type": "claims",
                "tenant_id": self.tenant_id,
                "token_usage": usage
            }

        except Exception as e:
            logger.error(f"Error in claims agent: {e}", exc_info=True)
            return {
                "content": "I apologize, but I encountered an error. Please try again or contact support.",
                "agent_type": "claims",
                "error": str(e),
            }
