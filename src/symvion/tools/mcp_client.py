"""
MCP (Model Context Protocol) client for calling MCP tools.
Handles communication with MCP servers via HTTP.
"""

import logging
import os
import httpx
from typing import Dict, Any, Optional

from symvion.utils.security import resolve_sandboxed_path, validate_outbound_url

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for interacting with MCP servers."""

    def __init__(
        self,
        server_url: str,
        server_name: str,
        *,
        allowed_outbound_hosts: Optional[list] = None,
        file_sandbox_dir: Optional[str] = None,
    ):
        """
        Initialize MCP client.

        Args:
            server_url: URL of the MCP server
            server_name: Name of the MCP server
            allowed_outbound_hosts: Optional host allowlist
            file_sandbox_dir: Directory under which local file_path reads are allowed
        """
        validated = validate_outbound_url(
            server_url,
            allowed_hosts=allowed_outbound_hosts,
            allow_localhost=True,
            require_https=False,
        )
        # Ensure URL ends with trailing slash to avoid redirects
        self.server_url = validated.rstrip("/") + "/"
        self.server_name = server_name
        self.file_sandbox_dir = file_sandbox_dir or os.environ.get("SYMVION_FILE_SANDBOX")
        # Never follow redirects: prevents SSRF / credential bounce.
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=False)

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
                    # Read file and send as base64 — only from sandbox
                    import base64

                    safe_path = resolve_sandboxed_path(
                        parameters["file_path"],
                        sandbox_dir=self.file_sandbox_dir,
                    )

                    with open(safe_path, "rb") as f:
                        file_content = f.read()

                    # Encode as base64 for transmission
                    file_base64 = base64.b64encode(file_content).decode("utf-8")

                    request_body = {
                        "file_data": file_base64,
                        "file_name": parameters.get("file_name", safe_path.name),
                        "mime_type": parameters.get("mime_type", ""),
                    }

                    response = await self.client.post(
                        f"{self.server_url}tools/extract_text",
                        json=request_body,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                    )
                else:
                    # Reject open remote download URLs unless host allowlist passes.
                    doc_url = parameters.get("document_url") or parameters.get("url")
                    if doc_url:
                        validate_outbound_url(
                            doc_url,
                            allow_localhost=False,
                            require_https=True,
                        )
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
            # Log status only — avoid dumping response bodies that may contain secrets.
            logger.error(
                "HTTP error calling MCP tool %s on %s: %s",
                tool_name,
                self.server_name,
                e.response.status_code,
            )
            raise
        except Exception as e:
            logger.error(
                "Error calling MCP tool %s on %s: %s",
                tool_name,
                self.server_name,
                type(e).__name__,
            )
            raise

    async def list_tools(self) -> list:
        """List available tools from the MCP server."""
        try:
            response = await self.client.get(f"{self.server_url}tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Error listing tools from %s: %s", self.server_name, type(e).__name__)
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


def get_mcp_client(
    server_name: str,
    *,
    allowed_outbound_hosts: Optional[list] = None,
    file_sandbox_dir: Optional[str] = None,
) -> Optional[MCPClient]:
    """
    Get an MCP client for a specific server.

    Args:
        server_name: Name of the MCP server
        allowed_outbound_hosts: Optional host allowlist
        file_sandbox_dir: Directory under which local file_path reads are allowed

    Returns:
        MCPClient instance or None if server not found
    """
    server_config = MCP_SERVERS.get(server_name)
    if not server_config:
        logger.warning("MCP server %s not found in configuration", server_name)
        return None

    return MCPClient(
        server_config["url"],
        server_name,
        allowed_outbound_hosts=allowed_outbound_hosts,
        file_sandbox_dir=file_sandbox_dir,
    )
