"""
Quote generation agent.
Handles insurance pricing and quotes.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from symvion.tools.quote_tools import get_quote_tools
from symvion.providers.factory import ProviderFactory
from symvion.agents.base import BaseAgent
from symvion.utils.helpers import ensure_messages, filter_allowed_tools
from symvion.core.context import TenantContext
from symvion.prompts.registry import PromptRegistry
from langchain_core.runnables import RunnableConfig
from symvion.tools.base import ToolSafetyWrapper

logger = logging.getLogger(__name__)

class QuoteAgent(BaseAgent):
    """Agent for generating insurance quotes."""

    def __init__(self, tenant_id: str, config: Dict[str, Any]):
        super().__init__(tenant_id, config)
        self.description = "Quotes specialist. Can calculate premiums for auto, life, or property insurance and save quote drafts."

        # Initialize tools with tenant-specific config
        backend_url = config.get("backend_url")
        auth_token = config.get("auth_token")
        self.quote_tools = get_quote_tools(tenant_id, backend_url, auth_token)
        
        self.tools = self._create_tools()
        
        self.tools = self._create_tools()
        # LLM is initialized in BaseAgent.__init__
        self.system_prompt = PromptRegistry.get_prompt("quote_agent", self.config)

    def _create_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.quote_tools.calculate_premium,
                name="calculate_premium",
                description="Calculate insurance premium. Need: coverage_type, coverage_amount, details(dict)."
            ),
            StructuredTool.from_function(
                func=self.quote_tools.save_quote,
                name="save_quote",
                description="Save a quote for a customer. Need: customer_email, quote_data(dict)."
            )
        ]

    def _build_system_prompt(self) -> str:
        """Construct the system prompt for the quote specialist."""
        return PromptRegistry.get_prompt("quote_agent", self.config)

    async def execute(self, context: TenantContext, input_data: Dict[str, Any], tools: Optional[Any] = None, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        message = input_data.get("message", "")
        history = input_data.get("history", [])
        
        # Use unified active governance to filter tools based on IAM policies
        allowed_tools = filter_allowed_tools(context, self.tools)
        llm_with_tools = self.llm.bind_tools(allowed_tools)
        
        # Build messages
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add history (ensure compatibility with standardized objects)
        messages.extend(ensure_messages(history[-5:]))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        # Invoke LLM using astream to enable token events
        response = None
        usage = {}
        async for chunk in llm_with_tools.astream(messages, config=config):
            if response is None:
                response = chunk
            else:
                response += chunk
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = chunk.usage_metadata
        
        tools_called = False
        if response.tool_calls:
            from symvion.tools.hitl import run_tool_with_hitl
            import json
            tools_called = True
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = dict(tool_call.get("args") or {})
                call_id = tool_call.get("id", f"call_{tool_name}")

                matched_tool = next((t for t in allowed_tools if t.name == tool_name), None)

                async def _exec(name=tool_name, **kwargs):
                    iam = context.metadata.get("iam_policies")
                    if name == "calculate_premium":
                        fn = self.quote_tools.calculate_premium
                        args = kwargs or {}
                    elif name == "save_quote":
                        fn = self.quote_tools.save_quote
                        args = dict(kwargs)
                        if "quote_data" not in args:
                            args["quote_data"] = {"note": "auto-filled for demo"}
                    else:
                        return f"Error: Tool {name} not found"
                    return await ToolSafetyWrapper.invoke(
                        fn, context, name, args, iam_policies=iam
                    )

                result = await run_tool_with_hitl(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    call_id=call_id,
                    execute=_exec,
                    config=config,
                    tool=matched_tool,
                )
                messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))
            
            # Get final response using astream
            response = None
            async for chunk in self.llm.astream(messages, config=config):
                if response is None:
                    response = chunk
                else:
                    response += chunk

        usage = response.usage_metadata if hasattr(response, "usage_metadata") else getattr(response, "response_metadata", {}).get("token_usage", {})
        
        return {
            "agent_response": response.content,
            "agent_type": "quote_generator",
            "token_usage": usage,
            "tools_called": tools_called 
        }
