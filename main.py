import os
from mcp.server.fastmcp import FastMCP
from predictif.tools import register_tools
from autotrain.mcp_tools import register_ml_tools

port = int(os.getenv("PORT", 3000))

mcp = FastMCP(
    name="Predictif ML Server",
    host="0.0.0.0",
    port=port,
    stateless_http=True,
    debug=False,
)

register_tools(mcp)
register_ml_tools(mcp)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
