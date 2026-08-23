"""
MCP (Model Context Protocol) client for calling MCP tools.
Handles communication with MCP servers via SSE (Server-Sent Events).
"""

import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for interacting with MCP servers."""

    def __init__(self, server_url: str, server_name: str):
        """
        Initialize MCP client.

        Args:
            server_url: URL of the MCP server
            server_name: Name of the MCP server
        """
        # Ensure URL ends with trailing slash to avoid redirects
        self.server_url = server_url.rstrip("/") + "/"
        self.server_name = server_name
        # Configure client to follow redirects (needed for POST requests that redirect)
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def call_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        try:
            # MCP tools via SSE use a different protocol
            # For document-processing, we need to call the extract_text tool
            # Try using the tool name in the URL path
            if tool_name == "document-processing":
                # For document processing, call the extract_text tool
                # Parameters can include file_path (local file) or document_url (remote URL)
                # If file_path is provided, we'll send the file content as base64
                if "file_path" in parameters:
                    # Read file and send as base64
                    import base64

                    file_path = parameters["file_path"]

                    # Read file content
                    with open(file_path, "rb") as f:
                        file_content = f.read()

                    # Encode as base64 for transmission
                    file_base64 = base64.b64encode(file_content).decode("utf-8")

                    # Try simpler request format - tool name in URL, parameters in body
                    request_body = {
                        "file_data": file_base64,
                        "file_name": parameters.get("file_name", ""),
                        "mime_type": parameters.get("mime_type", ""),
                    }

                    response = await self.client.post(
                        f"{self.server_url}tools/extract_text",  # Tool name in URL path
                        json=request_body,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                    )
                else:
                    # Fallback to URL-based approach
                    response = await self.client.post(
                        f"{self.server_url}tools/extract_text",
                        json=parameters,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                    )
            else:
                # Generic tool call - try tool name in URL
                response = await self.client.post(
                    f"{self.server_url}tools/{tool_name}",
                    json=parameters,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # Log response body for debugging
            try:
                error_body = (
                    e.response.text
                    if hasattr(e.response, "text")
                    else str(e.response.content)
                )
                logger.error(
                    f"HTTP error calling MCP tool {tool_name} on {self.server_name}: {e.response.status_code} - {error_body}"
                )
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(
                f"Error calling MCP tool {tool_name} on {self.server_name}: {e}",
                exc_info=True,
            )
            raise

    async def list_tools(self) -> list:
        """List available tools from the MCP server."""
        try:
            response = await self.client.get(f"{self.server_url}/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error listing tools from {self.server_name}: {e}")
            return []

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# MCP server configurations
MCP_SERVERS = {
    "ai.filegraph/document-processing": {
        "url": "https://api.filegraph.ai/mcp",
        "type": "sse",
    },
}


def get_mcp_client(server_name: str) -> Optional[MCPClient]:
    """
    Get an MCP client for a specific server.

    Args:
        server_name: Name of the MCP server

    Returns:
        MCPClient instance or None if server not found
    """
    server_config = MCP_SERVERS.get(server_name)
    if not server_config:
        logger.warning(f"MCP server {server_name} not found in configuration")
        return None

    return MCPClient(server_config["url"], server_name)
