"""
Basic tools for Predictif MCP Server
"""

from pydantic import Field
from mcp.server.fastmcp import FastMCP


def register_tools(mcp: FastMCP):
    """Register all tools with the MCP server"""

    @mcp.tool(
        title="Echo",
        description="Echo back the input message",
    )
    def echo(
        message: str = Field(description="Message to echo back")
    ) -> str:
        return f"Echo: {message}"