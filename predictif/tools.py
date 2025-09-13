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
        title="Save Document Text to File",
        description="Extracts text content from a document and saves it as a file in datasets/ directory. Use list_user_libraries() first to get library ID, then list_library_documents() to get document ID.",
    )
    def save_document_text_to_file(
        library_id: str = Field(
            description="ID of the library containing the document (get from list_user_libraries())"
        ),
        document_id: str = Field(
            description="ID of the document to save (get from list_library_documents())"
        ),
        custom_filename: str = Field(
            default="",
            description="Custom filename (optional, will use document name if not provided)",
        ),
    ) -> str:
        """
        Extracts text from a document and saves it as a file in datasets/ directory.

        Workflow:
        1. Call list_user_libraries() to find library ID from library name
        2. Call list_library_documents(library_id) to find document ID from document name
        3. Call this function with both IDs to save the text content as a file

        Args:
            library_id (str): The ID of the library containing the document
            document_id (str): The ID of the document to save
            custom_filename (str): Optional custom filename (will preserve original extension)

        Returns:
            str: Full path where the text file was saved
        """
        try:
            # Get document info to extract name and extension
            documents = mistral_client.beta.libraries.documents.list(
                library_id=library_id
            ).data

            document_name = None
            document_extension = None
            for doc in documents:
                if doc.id == document_id:
                    document_name = doc.name
                    document_extension = doc.extension
                    break

            if not document_name:
                return f"Error: Document with ID {document_id} not found in library {library_id}"

            # Extract text content from document
            extracted_text = mistral_client.beta.libraries.documents.text_content(
                library_id=library_id, document_id=document_id
            )
            text_content = extracted_text.text

            if not text_content.strip():
                return "Error: Document appears to be empty"

            # Generate filename
            if custom_filename:
                # If custom filename is provided, preserve original extension if it has one
                if "." in custom_filename:
                    filename = custom_filename
                else:
                    # Add original extension to custom filename
                    filename = (
                        f"{custom_filename}.{document_extension}"
                        if document_extension
                        else f"{custom_filename}.txt"
                    )
            else:
                # Use original document name
                filename = document_name

            # Create datasets directory if it doesn't exist
            datasets_dir = Path("datasets")
            datasets_dir.mkdir(exist_ok=True)

            # Full file path
            file_path = datasets_dir / filename

            # Check if file already exists
            if file_path.exists():
                return f"❌ File already exists at: datasets/{filename}\n📄 Source: {document_name}\nUse a different custom_filename to save with a new name."

            # Save text content to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            # Return relative path from project root
            relative_path = f"datasets/{filename}"

            return f"✅ Document saved successfully!\n📍 Dataset saved at: {relative_path}\n📄 Source: {document_name}\n📊 Size: {len(text_content)} characters"

        except Exception as e:
            return f"Error saving document {document_id} to file: {str(e)}"

    @mcp.tool(
        title="List Dataset Files",
        description="Lists all files in the datasets/ directory with detailed information including file size, type, and modification date.",
    )
    def list_dataset_files() -> str:
        """
        Lists all files stored in the datasets/ directory with comprehensive details.

        This function works independently - no prerequisites needed. Use it to:
        - Check what datasets are available before processing
        - Verify files were saved correctly after using save_document_text_to_file
        - Get file information for further analysis
        - Monitor dataset storage usage

        Returns:
            str: Formatted list of all files with details (name, size, type, modified date)
                 or message if directory is empty/doesn't exist
        """
        try:
            datasets_dir = Path("datasets")

            # Check if datasets directory exists
            if not datasets_dir.exists():
                return "📁 No datasets directory found. Use save_document_text_to_file to create it and add files."

            # Get all files in datasets directory
            files = list(datasets_dir.glob("*"))
            files = [f for f in files if f.is_file()]  # Only files, not directories

            if not files:
                return "📂 Datasets directory exists but is empty.\n💡 Use save_document_text_to_file to add files."

            # Sort files by modification time (newest first)
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Build detailed file listing
            result = []
            result.append("📊 Dataset Files")
            result.append("=" * 50)
            result.append(f"📁 Location: datasets/")
            result.append(f"📈 Total files: {len(files)}")
            result.append("")

            for file_path in files:
                stat = file_path.stat()
                file_size = stat.st_size

                # Format file size
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"

                # Get file extension for type
                file_type = file_path.suffix.lower() or "no extension"

                # Format modification time
                from datetime import datetime
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                mod_time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")

                result.append(f"📄 {file_path.name}")
                result.append(f"   └─ Size: {size_str}")
                result.append(f"   └─ Type: {file_type}")
                result.append(f"   └─ Modified: {mod_time_str}")
                result.append(f"   └─ Path: datasets/{file_path.name}")
                result.append("")

            # Add total size
            total_size = sum(f.stat().st_size for f in files)
            if total_size < 1024 * 1024:
                total_size_str = f"{total_size / 1024:.1f} KB"
            else:
                total_size_str = f"{total_size / (1024 * 1024):.1f} MB"

            result.append(f"💾 Total storage used: {total_size_str}")

            return "\n".join(result)

        except Exception as e:
            return f"Error listing dataset files: {str(e)}"

    @mcp.tool(
        title="Get model report",
        description="Get model report with detailed information",
    )
    def get_model_report(
        model_id: str = Field(description="ID of the created model"),
    ) -> str:
        """
        Generates the model report and returns it in the form of text.

        Args:
            model_id (str): id of the created model

        Returns:
            str: Clean, structured report with model information
        """
        base_dir = Path("models") / model_id
        results_path = base_dir / "metadata.json"
        model_path = base_dir / "model.pkl"  # currently unused

        if not base_dir.exists():
            return f"No directory found for model_id='{model_id}'"

        if not results_path.exists():
            return f"metadata.json not found in {base_dir}"

        try:
            with open(results_path, "r") as f:
                metadata: Dict = json.load(f)
        except Exception as e:
            return f"⚠️ Could not read metadata.json: {e}"

        if not isinstance(metadata, dict) or not metadata:
            return "⚠️ metadata.json is empty or not a valid JSON object."

        # Convert dict into rows for a table
        table_data = [(str(k), str(v)) for k, v in metadata.items()]
        table = tabulate(table_data, headers=["Property", "Value"], tablefmt="github")

        report = f"""
    📊 Model Report
    ====================
    **Model ID:** {model_id}

    📁 Files:
    - metadata.json ✅
    - model.pkl {"✅" if model_path.exists() else "❌"}

    {table}
    """
        return report
