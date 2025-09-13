from mcp.server.fastmcp import FastMCP
import pandas as pd
from pydantic import Field

mcp = FastMCP("Predictif MCP Server", port=3000, stateless_http=True, debug=True)


@mcp.tool(
    title="Echo Tool",
    description="Echo the input text",
)
def echo(text: str = Field(description="The text to echo")) -> str:
    return text


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
