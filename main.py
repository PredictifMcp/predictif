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
    host="0.0.0.0",  # Bind to all interfaces for Docker
    port=port,
    stateless_http=True,  # Better for containerized environments
    debug=False,  # Disable debug in production
)

# Register our tools
register_tools(mcp)

if __name__ == "__main__":
    # Run with SSE transport for better reverse proxy compatibility
    mcp.run(transport="sse")
