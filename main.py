"""
<<<<<<< HEAD
Predictif MCP Server - ML Model Training and Prediction Tools
=======
Prédictif MCP Server - ML Training and Inference
>>>>>>> d83c5ca (add MCP server)
"""

import os
from mcp.server.fastmcp import FastMCP
from predictif.tools import register_tools
from autotrain.mcp_tools import register_ml_tools

port = int(os.getenv("PORT", 3000))

mcp = FastMCP("Prédictif ML Server", port=3000, stateless_http=True, debug=True)

register_tools(mcp)
register_ml_tools(mcp)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
