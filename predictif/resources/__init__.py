"""Resources module for MCP server."""

import fastmcp

from .example_resources import register_example_resources


def setup_resources(app: fastmcp.FastMCP) -> None:
    """Set up all resources for the MCP server.

    Args:
        app: FastMCP application instance
    """
    register_example_resources(app)