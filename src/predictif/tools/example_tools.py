"""Example tools for the Predictif MCP server."""

import fastmcp
from ..utils.logging import get_logger

logger = get_logger(__name__)


def register_example_tools(app: fastmcp.FastMCP) -> None:
    """Register example tools with the MCP server.

    Args:
        app: FastMCP application instance
    """

    @app.tool()
    def hello_world(name: str = "World") -> str:
        """Say hello to someone.

        Args:
            name: Name to greet

        Returns:
            Greeting message
        """
        logger.info(f"Greeting {name}")
        return f"Hello, {name}!"

    @app.tool()
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of the numbers
        """
        logger.info(f"Adding {a} + {b}")
        return a + b
