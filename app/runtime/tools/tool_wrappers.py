"""
MCP tool call wrappers.
Provides secure abstractions for calling MCP tools with tenant isolation.
"""

import logging
from typing import Dict, Any, List, Optional
from app.runtime.tools.mcp_client import get_mcp_client, MCP_SERVERS

logger = logging.getLogger(__name__)


# Map tool IDs to their MCP server names
TOOL_TO_MCP_SERVER = {
    "document-processing": "ai.filegraph/document-processing",
}


class ToolWrapper:
    """Wrapper for MCP tool calls with tenant isolation."""

    def __init__(self, tenant_id: str, allowed_tools: List[str]):
        """
        Initialize tool wrapper.

        Args:
            tenant_id: Unique tenant identifier
            allowed_tools: List of tool identifiers allowed for this tenant
        """
        self.tenant_id = tenant_id
        self.allowed_tools = allowed_tools

    async def call_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP tool with validation and tenant isolation.

        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool is not in allowed_tools list
        """
        if tool_name not in self.allowed_tools:
            raise ValueError(
                f"Tool '{tool_name}' is not allowed for tenant '{self.tenant_id}'"
            )

        # Get MCP server for this tool
        mcp_server_name = TOOL_TO_MCP_SERVER.get(tool_name)
        if not mcp_server_name:
            raise ValueError(f"No MCP server configured for tool '{tool_name}'")

        # Get MCP client
        mcp_client = get_mcp_client(mcp_server_name)
        if not mcp_client:
            raise ValueError(f"Failed to initialize MCP client for tool '{tool_name}'")

        try:
            # Call the tool via MCP
            result = await mcp_client.call_tool(tool_name, parameters)
            logger.info(
                f"Successfully called tool '{tool_name}' for tenant '{self.tenant_id}'"
            )
            return result
        except Exception as e:
            logger.error(
                f"Error calling tool '{tool_name}' for tenant '{self.tenant_id}': {e}",
                exc_info=True,
            )
            raise
