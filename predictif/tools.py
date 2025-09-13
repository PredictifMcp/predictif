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
    def echo(message: str = Field(description="Message to echo back")) -> str:
        return f"Echo: {message}"

    @mcp.tool()
    def get_file_url(file_url: str) -> str:
        """
        Accepts a file URL parameter and prints it.
        This file parameter is the file_url of the file the user has posted inside the conversation

        Args:
            file_url (str): The file URL to process and print (file_url) the user has uploaded on the conversation

        Returns:
            str: Confirmation message with the processed URL
        """
        if not file_url:
            raise ValueError("file_url parameter is required and cannot be empty")

        # Print the file URL
        print(f"File URL: {file_url}")

        return f"File URL received and printed: {file_url}"
