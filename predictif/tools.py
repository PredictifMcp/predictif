"""
Basic tools for Predictif MCP Server
"""

import os
import io
import pandas as pd
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from mistralai import Mistral


def register_tools(mcp: FastMCP):
    """Register all tools with the MCP server"""

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is required")

    mistral_client = Mistral(api_key=mistral_api_key)

    @mcp.tool(
        title="Echo",
        description="Echo back the input message",
    )
    def echo(message: str = Field(description="Message to echo back")) -> str:
        return f"Echo: {message}"

    @mcp.tool(
        title="List User Libraries",
        description="Lists all libraries available to the current user with their document counts",
    )
    def list_user_libraries() -> str:
        """
        Lists all libraries available to the current user.

        Returns:
            str: Formatted list of libraries with their document counts
        """
        try:
            libraries = mistral_client.beta.libraries.list().data

            if not libraries:
                return "No libraries found for the current user."

            result = "User Libraries:\n"
            for library in libraries:
                result += f"- {library.name} with {library.nb_documents} documents with the following id : {library.id}\n"

            return result.strip()

        except Exception as e:
            return f"Error retrieving libraries: {str(e)}"

    @mcp.tool(
        title="List Library Documents",
        description="Lists all documents in a specific library with their details",
    )
    def list_library_documents(
        library_id: str = Field(description="ID of the library to list documents from"),
    ) -> str:
        """
        Lists all documents in a specific library.

        Args:
            library_id (str): The ID of the library to list documents from

        Returns:
            str: Formatted list of documents with their details
        """
        try:
            doc_list = mistral_client.beta.libraries.documents.list(
                library_id=library_id
            ).data

            if not doc_list:
                return f"No documents found in library with ID: {library_id}"

            result = f"Documents in library {library_id}:\n"
            for doc in doc_list:
                result += (
                    f"- {doc.name}: {doc.extension} with {doc.number_of_pages} pages\n"
                )
                if hasattr(doc, "summary") and doc.summary:
                    result += f"  Summary: {doc.summary}\n"

            return result.strip()

        except Exception as e:
            return f"Error retrieving documents from library {library_id}: {str(e)}"

    @mcp.tool(
        title="Extract Document Text",
        description="Extracts the full text content from a specific document in a library",
    )
    def extract_document_text(
        library_id: str = Field(
            description="ID of the library containing the document"
        ),
        document_id: str = Field(description="ID of the document to extract text from"),
    ) -> str:
        """
        Extracts the full text content from a document in a library.

        Args:
            library_id (str): The ID of the library containing the document
            document_id (str): The ID of the document to extract text from

        Returns:
            str: The extracted text content of the document
        """
        try:
            extracted_text = mistral_client.beta.libraries.documents.text_content(
                library_id=library_id, document_id=document_id
            )
            return extracted_text.text

        except Exception as e:
            return f"Error extracting text from document {document_id} in library {library_id}: {str(e)}"

    @mcp.tool(
        title="Analyze Document as CSV",
        description="Extracts text from a document, parses it as CSV using pandas, and provides a summary",
    )
    def analyze_document_as_csv(
        library_id: str = Field(
            description="ID of the library containing the document"
        ),
        document_id: str = Field(description="ID of the document to analyze as CSV"),
        separator: str = Field(
            default=",", description="CSV separator character (default: comma)"
        ),
    ) -> str:
        """
        Extracts text from a document, parses it as CSV, and provides a pandas summary.

        Args:
            library_id (str): The ID of the library containing the document
            document_id (str): The ID of the document to analyze as CSV
            separator (str): CSV separator character (default: comma)

        Returns:
            str: Pandas summary of the CSV data including shape, columns, data types, and basic statistics
        """
        try:
            # Extract text content from document
            extracted_text = mistral_client.beta.libraries.documents.text_content(
                library_id=library_id, document_id=document_id
            )
            text_content = extracted_text.text

            # Parse as CSV using pandas
            df = pd.read_csv(io.StringIO(text_content), sep=separator)

            # Generate comprehensive summary
            summary = []
            summary.append(f"CSV Analysis Summary:")
            summary.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            summary.append(f"")

            summary.append(f"Columns:")
            for i, col in enumerate(df.columns, 1):
                summary.append(f"  {i}. {col} ({df[col].dtype})")
            summary.append(f"")

            summary.append(f"Data Types:")
            for dtype in df.dtypes.value_counts().items():
                summary.append(f"  {dtype[0]}: {dtype[1]} columns")
            summary.append(f"")

            summary.append(f"Missing Values:")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                summary.append("  No missing values")
            else:
                for col, count in missing[missing > 0].items():
                    summary.append(f"  {col}: {count} missing")
            summary.append(f"")

            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                summary.append(f"Numeric Summary:")
                desc = df[numeric_cols].describe()
                summary.append(desc.to_string())
                summary.append(f"")

            # Sample data (first 5 rows)
            summary.append(f"Sample Data (first 5 rows):")
            summary.append(df.head().to_string())

            return "\n".join(summary)

        except pd.errors.EmptyDataError:
            return f"Error: Document appears to be empty or not a valid CSV"
        except pd.errors.ParserError as e:
            return (
                f"Error parsing CSV: {str(e)}. Try adjusting the separator parameter."
            )
        except Exception as e:
            return f"Error analyzing document {document_id} as CSV: {str(e)}"
