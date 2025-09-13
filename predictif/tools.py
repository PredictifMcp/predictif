"""
Basic tools for Predictif MCP Server
"""

import os
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from mistralai import Mistral


def register_tools(mcp: FastMCP):
    """Register all tools with the MCP server"""

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is required")

    mistral_client = Mistral(api_key=mistral_api_key)

    @mcp.tool(
        title="Echo",
        description="Echo back the input message",
    )
    def echo(message: str = Field(description="Message to echo back")) -> str:
        return f"Echo: {message}"

    @mcp.tool(
        title="List User Libraries",
        description="Lists all libraries available to the current user with their document counts",
    )
    def list_user_libraries() -> str:
        """
        Lists all libraries available to the current user.

        Returns:
            str: Formatted list of libraries with their document counts
        """
        try:
            libraries = mistral_client.beta.libraries.list().data

            if not libraries:
                return "No libraries found for the current user."

            result = "User Libraries:\n"
            for library in libraries:
                result += f"- {library.name} with {library.nb_documents} documents\n"

            return result.strip()

        except Exception as e:
            return f"Error retrieving libraries: {str(e)}"
