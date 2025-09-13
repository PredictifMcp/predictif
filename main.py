import os
from mcp.server.fastmcp import FastMCP
from predictif.tools import register_tools
from mltools.mcp_tools import register_ml_tools

port = int(os.getenv("PORT", 3000))

mcp = FastMCP(
    name="Predictif ML Server",
    host="0.0.0.0",
    port=port,
    stateless_http=True,
    debug=False,
)


@mcp.resource("predictif://usage-guide")
def usage_guide() -> str:
    """Guide for using Predictif MCP Server tools effectively"""
    return """
# Predictif MCP Server Usage Guide

## Document and Library Management Workflow

### 1. Basic Workflow for Document Operations

When working with documents, always follow this sequence:

1. **List Available Libraries**:
   - Use `list_user_libraries()` first
   - This returns structured data: `[Library Name] -> ID: [library_id] | Documents: [count]`
   - Find your target library and note its ID

2. **List Documents in Library**:
   - Use `list_library_documents(library_id)` with the ID from step 1
   - This returns: `[Document Name] -> ID: [document_id] | Type: [extension]`
   - Find your target document and note its ID

3. **Work with Specific Document**:
   - Use `extract_document_text(library_id, document_id)` to get text content
   - Or use `analyze_document_as_csv(library_id, document_id)` for CSV analysis

### 2. Example Scenarios

**Scenario: "List documents in library dataset"**
1. Call `list_user_libraries()` to find library named "dataset"
2. Extract the library ID from the structured output
3. Call `list_library_documents(library_id)` with that ID
4. Get clean list of document names with their IDs

**Scenario: "Analyze CSV file named 'sales.csv' in 'reports' library"**
1. Call `list_user_libraries()` to find "reports" library ID
2. Call `list_library_documents(library_id)` to find "sales.csv" document ID
3. Call `analyze_document_as_csv(library_id, document_id)`

### 3. Output Format Benefits

- **Structured Output**: Easy to parse name-to-ID mappings
- **No Unnecessary Descriptions**: Clean, focused information
- **Clear Workflow**: Functions guide you through the proper sequence
- **Error Prevention**: Descriptive parameter hints prevent common mistakes

### 4. Best Practices

- Always start with `list_user_libraries()` when you need to work with a library by name
- Use the structured output format to programmatically extract IDs
- Follow the suggested workflow in function descriptions
- Library and document IDs are required for all document operations
"""


@mcp.resource("predictif://api-reference")
def api_reference() -> str:
    """Quick reference for all available tools"""
    return """
# Predictif MCP Server API Reference

## Core Library Tools

### list_user_libraries()
- **Purpose**: Get all available libraries with name-to-ID mapping
- **Returns**: Structured list: `[Library Name] -> ID: [library_id] | Documents: [count]`
- **Use Case**: First step when working with libraries by name

### list_library_documents(library_id)
- **Purpose**: Get all documents in a library with name-to-ID mapping
- **Parameters**: `library_id` (from list_user_libraries)
- **Returns**: Structured list: `[Document Name] -> ID: [document_id] | Type: [extension]`
- **Use Case**: Second step to find specific documents

### extract_document_text(library_id, document_id)
- **Purpose**: Get full text content of a document
- **Parameters**: Both IDs from previous functions
- **Returns**: Raw text content
- **Use Case**: Text analysis, content review

### analyze_document_as_csv(library_id, document_id, separator=",")
- **Purpose**: Parse document as CSV and provide pandas summary
- **Parameters**: Both IDs plus optional separator
- **Returns**: Comprehensive CSV analysis (shape, columns, types, statistics)
- **Use Case**: Data analysis, CSV validation

## Utility Tools

### echo(message)
- **Purpose**: Simple echo for testing
- **Parameters**: `message` (string)
- **Returns**: "Echo: {message}"
- **Use Case**: Server connectivity testing
"""


register_tools(mcp)
register_ml_tools(mcp)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
