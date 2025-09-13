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

    @mcp.tool(
        title="List Library Documents",
        description="Lists all documents in a specific library with their details",
    )
    def list_library_documents(library_id: str = Field(description="ID of the library to list documents from")) -> str:
        """
        Lists all documents in a specific library.

        Args:
            library_id (str): The ID of the library to list documents from

        Returns:
            str: Formatted list of documents with their details
        """
        try:
            doc_list = mistral_client.beta.libraries.documents.list(library_id=library_id).data

            if not doc_list:
                return f"No documents found in library with ID: {library_id}"

            result = f"Documents in library {library_id}:\n"
            for doc in doc_list:
                result += f"- {doc.name}: {doc.extension} with {doc.number_of_pages} pages\n"
                if hasattr(doc, 'summary') and doc.summary:
                    result += f"  Summary: {doc.summary}\n"

            return result.strip()

        except Exception as e:
            return f"Error retrieving documents from library {library_id}: {str(e)}"
