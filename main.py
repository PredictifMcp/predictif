"""
Predictif MCP Server - ML Model Training and Prediction Tools
"""

import os
from mcp.server.fastmcp import FastMCP
from predictif.tools import register_tools

# Get port from environment variable or default to 3000
port = int(os.getenv("PORT", 3000))

mcp = FastMCP(
    name="Predictif ML Server",
    host="0.0.0.0",
    port=port,
    stateless_http=True,
    debug=False,
)

register_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="sse")
