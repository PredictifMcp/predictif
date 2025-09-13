"""Example resources for the Predictif MCP server."""

import fastmcp
from ..utils.logging import get_logger

logger = get_logger(__name__)


def register_example_resources(app: fastmcp.FastMCP) -> None:
    """Register example resources with the MCP server.

    Args:
        app: FastMCP application instance
    """

    @app.resource("config://server")
    def get_server_config() -> str:
        """Get server configuration information."""
        logger.info("Fetching server configuration")
        return "Server configuration: Predictif MCP Server v0.1.0"

    @app.resource("data://sample")
    def get_sample_data() -> str:
        """Get sample data."""
        logger.info("Fetching sample data")
        return "Sample data: [1, 2, 3, 4, 5]"