"""Main entry point for the Predictif MCP server."""

import asyncio
from typing import Any

import fastmcp

from .tools import setup_tools
from .resources import setup_resources
from .utils.logging import setup_logging


app = fastmcp.FastMCP("Predictif", port=3001, stateless_http=True, debug=True)


def setup_server() -> None:
    setup_logging()
    setup_tools(app)
    setup_resources(app)


def main() -> None:
    setup_server()
    app.run(transport="streamable-http")


if __name__ == "__main__":
    main()
