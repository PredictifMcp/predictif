"""Tools module for MCP server."""

import fastmcp

from .example_tools import register_example_tools


def setup_tools(app: fastmcp.FastMCP) -> None:
    """Set up all tools for the MCP server.

    Args:
        app: FastMCP application instance
    """
    register_example_tools(app)