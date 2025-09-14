import os
from mcp.server.fastmcp import FastMCP
from predictif.ml_tools import register_ml_tools
from predictif.files_tools import register_file_tools

port = int(os.getenv("PORT", 3000))

mcp = FastMCP(
    name="Predictif ML Server",
    host="0.0.0.0",
    port=port,
    stateless_http=True,
    debug=False,
)

register_ml_tools(mcp)
register_file_tools(mcp)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
