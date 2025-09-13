from fastmcp import FastMCP
from mcp.server import Server
import asyncio

mcp = FastMCP("predictif-server")

@mcp.tool()
def hello_world(name: str) -> str:
    """A simple hello world function."""
    return f"Hello, {name}!"

@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

async def main():
    async with mcp.run_server() as (read_stream, write_stream):
        await Server(mcp.create_server()).run(
            read_stream, write_stream, mcp.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
