from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from pydantic import Field
from mcp.server.fastmcp import FastMCP

# Create Flask app with CORS
app = Flask(__name__)
CORS(app)

# Create FastMCP server
mcp = FastMCP(
    name="Predictif MCP Server",
    host="0.0.0.0",
    port=3019,
)

# Add Flask routes for health checks
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "Predictif MCP Server"})

@mcp.tool(
    title="Echo Tool",
    description="Echo the input text",
)
def echo(text: str = Field(description="The text to echo")) -> str:
    return text

if __name__ == "__main__":
    # Run MCP server with SSE transport for Le Chat
    mcp.run(transport="sse")
