"""
MCP tools for file and library operations
"""

import pandas as pd
from pydantic import Field
from mcp.server.fastmcp import FastMCP

from .files import FileManager


def register_file_tools(mcp: FastMCP):
    file_manager = FileManager()

    @mcp.tool(
        title="Get Library File",
        description="Find and get information about a specific file by name across all libraries.",
    )
    def get_library_file(
        filename: str = Field(description="Name of the file to find (e.g., 'data.csv', 'report.pdf')")
    ) -> str:
        try:
            file_info = file_manager.find_file(filename)
            if not file_info:
                return f"File '{filename}' not found in any library."

            return f"Found '{filename}' in library '{file_info['library_name']}' (ID: {file_info['library_id']}) - Document ID: {file_info['document_id']} - Type: {file_info['document_type']}"
        except Exception as e:
            return f"Error searching for file {filename}: {str(e)}"

    @mcp.tool(
        title="Extract Document Text",
        description="Extracts the full text content from a document by filename.",
    )
    def extract_document_text(
        filename: str = Field(description="Name of the file to extract text from"),
    ) -> str:
        try:
            file_info = file_manager.find_file(filename)
            if not file_info:
                return f"File '{filename}' not found in any library."

            return file_manager.extract_text(file_info['library_id'], file_info['document_id'])
        except Exception as e:
            return f"Error extracting text from {filename}: {str(e)}"

    @mcp.tool(
        title="Analyze Document as CSV",
        description="Analyzes a document as CSV by filename.",
    )
    def analyze_document_as_csv(
        filename: str = Field(description="Name of the CSV file to analyze"),
        separator: str = Field(default=",", description="CSV separator character"),
    ) -> str:
        try:
            file_info = file_manager.find_file(filename)
            if not file_info:
                return f"File '{filename}' not found in any library."

            return file_manager.analyze_csv(file_info['library_id'], file_info['document_id'], separator)
        except pd.errors.EmptyDataError:
            return "Error: Document appears to be empty or not a valid CSV"
        except pd.errors.ParserError as e:
            return f"Error parsing CSV: {str(e)}. Try adjusting the separator parameter."
        except Exception as e:
            return f"Error analyzing {filename} as CSV: {str(e)}"

    @mcp.tool(
        title="Save Document Text to File",
        description="Saves a document to datasets/ directory by filename.",
    )
    def save_document_text_to_file(
        filename: str = Field(description="Name of the file to save"),
    ) -> str:
        try:
            file_info = file_manager.find_file(filename)
            if not file_info:
                return f"File '{filename}' not found in any library."

            return file_manager.save_document(file_info['library_id'], file_info['document_id'])
        except Exception as e:
            return f"Error saving {filename} to file: {str(e)}"

    @mcp.tool(
        title="List Datasets",
        description="List all saved datasets in the datasets/ directory.",
    )
    def list_datasets() -> str:
        try:
            return file_manager.list_datasets()
        except Exception as e:
            return f"Error listing datasets: {str(e)}"

    @mcp.tool(
        title="Delete Dataset",
        description="Delete a saved dataset file by filename.",
    )
    def delete_dataset(
        filename: str = Field(description="Name of the dataset file to delete"),
    ) -> str:
        try:
            return file_manager.delete_dataset(filename)
        except Exception as e:
            return f"Error deleting dataset: {str(e)}"