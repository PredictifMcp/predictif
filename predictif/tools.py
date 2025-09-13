"""
Basic tools for Predictif MCP Server
"""

import os
import io
import pandas as pd
import json
from pathlib import Path
from pydantic import Field
from typing import Dict
from mcp.server.fastmcp import FastMCP
from mistralai import Mistral
from tabulate import tabulate 


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
        description="Lists all libraries available to the current user. Returns structured data mapping library names to IDs and document counts. This should be called before listing documents in a specific library to get the library ID from the library name.",
    )
    def list_user_libraries() -> str:
        """
        Lists all libraries available to the current user with structured output for easy name-to-ID mapping.
        Use this function first when you need to find a library by name, then use the returned ID
        with list_library_documents().

        Returns:
            str: Structured list of libraries with clear name-to-ID mapping
        """
        try:
            libraries = mistral_client.beta.libraries.list().data

            if not libraries:
                return "No libraries found for the current user."

            result = "Available Libraries:\n"
            result += (
                "Format: [Library Name] -> ID: [library_id] | Documents: [count]\n\n"
            )

            for library in libraries:
                result += f"[{library.name}] -> ID: {library.id} | Documents: {library.nb_documents}\n"

            return result.strip()

        except Exception as e:
            return f"Error retrieving libraries: {str(e)}"

    @mcp.tool(
        title="List Library Documents",
        description="Lists all documents in a specific library with structured output. Use list_user_libraries() first to get the library ID from a library name, then use that ID with this function.",
    )
    def list_library_documents(
        library_id: str = Field(
            description="ID of the library to list documents from (get this from list_user_libraries() first)"
        ),
    ) -> str:
        """
        Lists all documents in a specific library with clean, structured output optimized for name-to-ID mapping.

        Workflow:
        1. First call list_user_libraries() to find the library ID from the library name
        2. Then call this function with the library ID

        Args:
            library_id (str): The ID of the library to list documents from

        Returns:
            str: Clean, structured list of documents with name-to-ID mapping
        """
        try:
            doc_list = mistral_client.beta.libraries.documents.list(
                library_id=library_id
            ).data

            if not doc_list:
                return f"No documents found in library with ID: {library_id}"

            result = f"Documents in Library {library_id}:\n"
            result += (
                "Format: [Document Name] -> ID: [document_id] | Type: [extension]\n\n"
            )

            for doc in doc_list:
                result += f"[{doc.name}] -> ID: {doc.id} | Type: {doc.extension}\n"

            return result.strip()

        except Exception as e:
            return f"Error retrieving documents from library {library_id}: {str(e)}"

    @mcp.tool(
        title="Extract Document Text",
        description="Extracts the full text content from a specific document in a library. Use list_user_libraries() first to get library ID, then list_library_documents() to get document ID.",
    )
    def extract_document_text(
        library_id: str = Field(
            description="ID of the library containing the document (get from list_user_libraries())"
        ),
        document_id: str = Field(
            description="ID of the document to extract text from (get from list_library_documents())"
        ),
    ) -> str:
        """
        Extracts the full text content from a document in a library.

        Workflow:
        1. Call list_user_libraries() to find library ID from library name
        2. Call list_library_documents(library_id) to find document ID from document name
        3. Call this function with both IDs

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
        description="Extracts text from a document, parses it as CSV using pandas, and provides a summary. Use list_user_libraries() first to get library ID, then list_library_documents() to get document ID.",
    )
    def analyze_document_as_csv(
        library_id: str = Field(
            description="ID of the library containing the document (get from list_user_libraries())"
        ),
        document_id: str = Field(
            description="ID of the document to analyze as CSV (get from list_library_documents())"
        ),
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

    @mcp.tool(
        title="Get Document Signed URLs",
        description="Retrieves secure, time-limited signed URLs for accessing document content. Use list_user_libraries() first to get library ID, then list_library_documents() to get document ID.",
    )
    def get_document_signed_urls(
        library_id: str = Field(
            description="ID of the library containing the document (get from list_user_libraries())"
        ),
        document_id: str = Field(
            description="ID of the document to get signed URLs for (get from list_library_documents())"
        ),
    ) -> str:
        """
        Generates secure, time-limited signed URLs for document access using Mistral's API.

        This function uses the Mistral API client's `extracted_text_signed_url()` method to:
        - Generate a signed URL for accessing the document's extracted text content
        - Generate a signed URL for downloading the original raw document file
        - Provide secure, temporary access without exposing permanent URLs

        Workflow:
        1. Call list_user_libraries() to find library ID from library name
        2. Call list_library_documents(library_id) to find document ID from document name
        3. Call this function with both IDs to get secure access URLs

        Args:
            library_id (str): The ID of the library containing the document
            document_id (str): The ID of the document to generate signed URLs for

        Returns:
            str: Structured output containing signed URLs and usage instructions
        """
        try:
            # Use Mistral API client's extracted_text_signed_url method
            # This method returns both text content and raw document signed URLs
            signed_urls_response = mistral_client.beta.libraries.documents.extracted_text_signed_url(
                library_id=library_id,
                document_id=document_id
            )

            # Build structured response
            result = []
            result.append(f"Signed URLs for Document {document_id}:")
            result.append("=" * 50)
            result.append("")

            # Display available signed URLs from the API response
            urls_found = False

            if hasattr(signed_urls_response, 'signed_url') and signed_urls_response.signed_url:
                result.append("🔗 Extracted Text Content URL:")
                result.append(f"  {signed_urls_response.signed_url}")
                result.append("  ↳ Access the processed, extracted text content from the document")
                result.append("")
                urls_found = True

            if hasattr(signed_urls_response, 'raw_signed_url') and signed_urls_response.raw_signed_url:
                result.append("📁 Raw Document Download URL:")
                result.append(f"  {signed_urls_response.raw_signed_url}")
                result.append("  ↳ Download the original document file in its native format")
                result.append("")
                urls_found = True

            if not urls_found:
                result.append("❌ No signed URLs available for this document")
                result.append("")

            # Usage information
            result.append("📋 Important Notes:")
            result.append("  • These URLs are time-limited and will expire")
            result.append("  • URLs provide secure, temporary access to document content")
            result.append("  • Text Content URL: Returns processed text (ideal for analysis)")
            result.append("  • Raw Document URL: Returns original file (ideal for download)")
            result.append("  • Use these URLs directly in HTTP requests, browsers, or applications")

            return "\n".join(result)

        except Exception as e:
            return f"Error generating signed URLs for document {document_id} in library {library_id}: {str(e)}"
            
    @mcp.tool(
        title="Get model report",
        description="Get model report with detailed information",
    )
    def get_model_report(
        model_id: str = Field(
            description="ID of the created model"
        )
    ) -> str:
        """
        Generates the model report and returns it in the form of text.

        Args:
            model_id (str): id of the created model

        Returns:
            str: Clean, structured report with model information
        """
        base_dir = Path("models") / model_id
        results_path = base_dir / "results.json"
        model_path = base_dir / "model.pkl"  # currently unused

        if not base_dir.exists():
            return f"No directory found for model_id='{model_id}'"

        if not results_path.exists():
            return f"results.json not found in {base_dir}"

        try:
            with open(results_path, "r") as f:
                metadata: Dict = json.load(f)
        except Exception as e:
            return f"⚠️ Could not read results.json: {e}"

        if not isinstance(metadata, dict) or not metadata:
            return "⚠️ results.json is empty or not a valid JSON object."

        # Convert dict into rows for a table
        table_data = [(str(k), str(v)) for k, v in metadata.items()]
        table = tabulate(table_data, headers=["Property", "Value"], tablefmt="github")

        report = f"""
    📊 Model Report
    ====================
    **Model ID:** {model_id}

    📁 Files:
    - results.json ✅
    - model.pkl {'✅' if model_path.exists() else '❌'}

    {table}
    """
        return report

