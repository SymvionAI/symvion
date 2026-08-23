"""
Claims processing agent.
Handles insurance claims workflows and interactions.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from app.runtime.tools.claims_tools import get_claims_tools
from app.runtime.agents.fraud_scoring_agent import FraudScoringAgent

logger = logging.getLogger(__name__)

# Roles allowed to use the fraud scoring sub-agent (not customers)
FRAUD_SCORING_ALLOWED_ROLES = frozenset({"super_admin", "tenant_admin", "human_agent"})


class ClaimsAgent:
    """Agent for processing insurance claims."""

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        """
        Initialize claims agent.

        Args:
            tenant_id: Unique tenant identifier
            config: Agent-specific configuration
        """
        self.tenant_id = tenant_id
        self.config = config

        # Initialize claims tools
        backend_url = config.get("backend_url") or os.getenv("BACKEND_URL")
        auth_token = config.get("auth_token")  # Get auth token from config
        self.claims_tools = get_claims_tools(tenant_id, backend_url, auth_token)

        # Fraud scoring sub-agent (used via get_fraud_score tool)
        self.fraud_scoring_agent = FraudScoringAgent(
            tenant_id, config.get("fraud_scoring") or {}
        )

        # Create LangChain tools from claims tools (includes fraud scoring)
        self.tools = self._create_tools()

        # Initialize LLM with tools bound
        model_name = config.get("model", os.getenv("LLM_MODEL", "gpt-4"))
        temperature = config.get(
            "temperature", float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _create_tools(self) -> List[StructuredTool]:
        """Create LangChain tools from claims tools."""
        return [
            StructuredTool.from_function(
                func=self.claims_tools.submit_claim,
                name="submit_claim",
                description="Submit a new insurance claim. Use this when a customer wants to file a new claim. Requires policy_number, incident_date, incident_description, and claim_type.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.get_claim_status,
                name="get_claim_status",
                description="Get the current status of an existing claim. Use this when a customer asks about their claim status or wants to check on a claim.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.get_claim_details,
                name="get_claim_details",
                description="Get detailed information about a specific claim. Use this when a customer wants full details about their claim.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.update_claim,
                name="update_claim",
                description="Update an existing claim with new information. Use this when a customer needs to modify their claim or provide additional information.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.list_claims,
                name="list_claims",
                description="List claims for a customer. Use this when a customer wants to see all their claims or filter by policy, status, or type.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.upload_claim_document,
                name="upload_claim_document",
                description="Attach a document to a claim. Use this when a customer wants to upload supporting documents like receipts, photos, or reports. Requires document_url from document upload.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.calculate_claim_estimate,
                name="calculate_claim_estimate",
                description="Calculate an estimated claim amount based on incident details. Use this to help customers understand potential claim values before submitting.",
            ),
            StructuredTool.from_function(
                func=self.claims_tools.get_claim_timeline,
                name="get_claim_timeline",
                description="Get the timeline and status history of a claim. Use this when a customer wants to see the progress and history of their claim processing.",
            ),
            StructuredTool.from_function(
                func=self._get_fraud_score,
                name="get_fraud_score",
                description="Get a fraud risk assessment for a claim. Use when the user asks about fraud risk or after submitting a claim. Pass claim details when available.",
            ),
        ]

    def _build_system_prompt(self) -> str:
        """Build the system prompt for claims agent."""
        return f"""You are a knowledgeable insurance claims assistant for tenant {self.tenant_id}.

Your role is to:
- Help customers file new insurance claims
- Check the status of existing claims
- Answer questions about claims processes and requirements
- Guide customers through the claims documentation process
- Explain claim policies, coverage, and procedures
- Assist with claim-related inquiries

You have access to tools that allow you to:
- Submit new claims (submit_claim) - REQUIRES: policy_number, incident_date (YYYY-MM-DD format), incident_description, claim_type
- Check claim status (get_claim_status)
- Get claim details (get_claim_details)
- Update claims (update_claim)
- List customer claims (list_claims)
- Upload documents to claims (upload_claim_document)
- Calculate claim estimates (calculate_claim_estimate)
- View claim timelines (get_claim_timeline)
- Get fraud risk assessment (get_fraud_score) - optional; use when user asks about fraud risk or after submitting a claim

FILING A NEW CLAIM - CRITICAL RULES:
1. Do NOT call submit_claim until you have collected ALL of the following from the user via clarifying questions:
   - policy_number (ask: "What is your policy number?")
   - incident_date in YYYY-MM-DD format (ask: "When did the incident occur? Please give the date (e.g. 2024-01-15)." If they give natural language like "last Tuesday", convert to YYYY-MM-DD)
   - incident_description (ask: "Can you describe what happened?")
   - claim_type (ask: "What type of claim is this? (e.g. auto, property, health, life)")
2. Ask these questions in a clear, conversational way. You may ask for one or two at a time, then ask for the rest. Do not submit until you have all four.
3. Do NOT offer to transfer the user to another agent or team member to file the claim. You are the claims assistant; you file the claim once you have the required information.
4. After you successfully submit a claim via submit_claim, you MUST tell the user that a member of their team will reach out (e.g. "Your claim has been submitted. A member of our team will reach out to you shortly to discuss next steps." or similar).

When helping customers with other tasks (status, details, list, etc.):
- Use the appropriate tools to access real claim data
- Ask for necessary information (claim number, policy number) before using tools when needed
- Provide clear instructions on required documentation
- Explain timelines and next steps based on actual claim status
- Be professional, empathetic, and thorough
- If a tool call fails, explain the error clearly and suggest alternatives

Always use tools when you need to access or modify claim data. Don't make up claim information.

You have a fraud scoring sub-agent available via get_fraud_score. Use it when:
- The user asks about fraud risk, whether their claim might be flagged, or similar
- After submitting a claim, you may optionally get a fraud risk assessment to inform the user (e.g. "Your claim has been submitted. Our initial risk assessment is low/medium/high; here's a brief summary.")
Pass whatever claim details you have (incident_description, estimated_amount, claim_type, policy_number, incident_date) to get a more accurate score."""

    async def _get_fraud_score(
        self,
        incident_description: str = "",
        estimated_amount: Optional[float] = None,
        claim_type: str = "",
        policy_number: str = "",
        incident_date: str = "",
        user_message: str = "",
    ) -> Dict[str, Any]:
        """Invoke the fraud scoring sub-agent and return the assessment.

        Returns a dict with: score (0-100), level (low/medium/high), factors (list), summary (str).
        Use when the user asks about fraud risk or after submitting a claim; pass whatever
        claim details you have so the assessment is more accurate.

        Args:
            incident_description: Free-text description of the incident (what happened, where, when).
            estimated_amount: Claim amount or estimate in currency units (e.g. 5000.00). Omit if unknown.
            claim_type: Type of claim (e.g. auto, property, health, theft).
            policy_number: Policy number or identifier if the customer provided it.
            incident_date: Date of the incident in YYYY-MM-DD format (e.g. 2024-01-15).
            user_message: Optional; the customer's exact or recent message for extra context.
        """
        return await self.fraud_scoring_agent.score(
            incident_description=incident_description or None,
            estimated_amount=estimated_amount,
            claim_type=claim_type or None,
            policy_number=policy_number or None,
            incident_date=incident_date or None,
            user_message=user_message or None,
        )

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process claims-related input with tool support.

        Args:
            input_data: Input data containing:
                - message: User's message content
                - conversation_id: Conversation identifier
                - context: Additional context
                - history: Previous conversation messages

        Returns:
            Dict with 'content' or 'response' key containing the agent's response
        """
        try:
            message = input_data.get("message", "")
            history = input_data.get("history", [])
            context = input_data.get("context", {})

            # Fraud scoring tool is only for super_admin, tenant_admin, human_agent
            user_role = (context.get("user_role") or "").strip().lower()
            allowed_tools = (
                self.tools
                if user_role in FRAUD_SCORING_ALLOWED_ROLES
                else [t for t in self.tools if t.name != "get_fraud_score"]
            )
            if user_role and user_role not in FRAUD_SCORING_ALLOWED_ROLES:
                logger.debug(
                    "Claims agent: user_role=%s not in allowed roles; fraud scoring tool disabled",
                    user_role or "(none)",
                )
            llm_with_tools = self.llm.bind_tools(allowed_tools)

            # Build conversation history
            messages = [SystemMessage(content=self.system_prompt)]

            # If document intelligence passed a document analysis (e.g. after upload), inject it
            document_analysis = context.get("document_analysis")
            if document_analysis:
                messages.append(
                    SystemMessage(
                        content=(
                            "The user uploaded a document that was already analyzed. You have the following document intelligence summary (use it as the source of truth for this conversation):\n\n"
                            f"{document_analysis}\n\n"
                            "RULES: (1) Acknowledge the document and offer to file the claim. (2) When the user says the information is in the document (e.g. 'it\'s in the document', 'you have it', 'it\'s above'), you MUST extract policy_number, incident_date (YYYY-MM-DD), incident_description, and claim_type FROM the summary above and use them to call submit_claim. Do NOT say you cannot read the document—you have the summary. (3) If the summary does not contain a clear date, policy number, or description, ask only for the missing piece(s)."
                        )
                    )
                )

            # Add conversation history
            for hist_msg in history[-10:]:  # Last 10 messages for context
                if hist_msg.get("role") == "user":
                    messages.append(HumanMessage(content=hist_msg.get("content", "")))
                elif hist_msg.get("role") == "assistant":
                    messages.append(AIMessage(content=hist_msg.get("content", "")))

            # Add current user message
            messages.append(HumanMessage(content=message))

            # Get response from LLM with tools (role-filtered)
            response = await llm_with_tools.ainvoke(messages)

            # Check if LLM wants to call tools
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})

                    logger.info(
                        f"Claims agent calling tool: {tool_name} with args: {tool_args}"
                    )

                    # Find the tool and execute it (only tools in allowed_tools were exposed)
                    tool_func = None
                    for tool in allowed_tools:
                        if tool.name == tool_name:
                            tool_func = tool.func
                            break
                    if not tool_func and tool_name == "get_fraud_score":
                        logger.warning(
                            "Claims agent: get_fraud_score called but not allowed for user_role=%s",
                            user_role or "(none)",
                        )
                        tool_results.append(
                            {"tool": tool_name, "error": "Fraud scoring is not available for this user."}
                        )
                        continue

                    if tool_func:
                        try:
                            result = await tool_func(**tool_args)
                            tool_results.append(
                                {
                                    "tool": tool_name,
                                    "result": result,
                                }
                            )
                        except Exception as tool_error:
                            logger.error(
                                f"Error executing tool {tool_name}: {tool_error}",
                                exc_info=True,
                            )
                            tool_results.append(
                                {
                                    "tool": tool_name,
                                    "error": str(tool_error),
                                }
                            )
                    else:
                        logger.warning(f"Tool {tool_name} not found")
                        tool_results.append(
                            {
                                "tool": tool_name,
                                "error": "Tool not found",
                            }
                        )

                # Add tool results to messages and get final response
                # Add the AI response with tool calls to messages first (required for proper message format)
                messages.append(response)

                # Match tool results with tool calls
                for i, tool_call in enumerate(response.tool_calls):
                    tool_result = (
                        tool_results[i]
                        if i < len(tool_results)
                        else {"error": "No result"}
                    )
                    tool_call_id = tool_call.get("id", f"call_{i}")

                    if "error" in tool_result:
                        messages.append(
                            ToolMessage(
                                content=f"Error: {tool_result['error']}",
                                tool_call_id=tool_call_id,
                            )
                        )
                    else:
                        # Convert result to JSON string for better formatting
                        import json

                        result_str = (
                            json.dumps(tool_result["result"])
                            if isinstance(tool_result["result"], (dict, list))
                            else str(tool_result["result"])
                        )
                        messages.append(
                            ToolMessage(
                                content=result_str,
                                tool_call_id=tool_call_id,
                            )
                        )

                # Get final response after tool execution
                final_response = await self.llm.ainvoke(messages)
                response_content = final_response.content
            else:
                # No tools called, use direct response
                response_content = response.content

            logger.debug(
                f"Claims agent response for tenant {self.tenant_id}: {response_content[:100]}"
            )

            return {
                "content": response_content,
                "agent_type": "claims",
                "tenant_id": self.tenant_id,
            }

        except Exception as e:
            logger.error(f"Error in claims agent: {e}", exc_info=True)
            return {
                "content": "I apologize, but I encountered an error while processing your claims inquiry. Please try again or contact support.",
                "agent_type": "claims",
                "error": str(e),
            }
