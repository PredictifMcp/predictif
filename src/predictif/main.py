"""Main entry point for the Predictif MCP server."""

import asyncio
from typing import Any

import fastmcp

from .tools import setup_tools
from .resources import setup_resources
from .utils.logging import setup_logging


app = fastmcp.FastMCP("Predictif")


def setup_server() -> None:
    """Set up the MCP server with tools and resources."""
    setup_logging()
    setup_tools(app)
    setup_resources(app)


def main() -> None:
    """Main entry point."""
    setup_server()
    fastmcp.run_stdio(app)


if __name__ == "__main__":
    main()