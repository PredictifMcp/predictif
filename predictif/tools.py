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
        description="Extracts text content from a document and saves it as a file in datasets/ directory with proper validation and path handling. Use list_user_libraries() first to get library ID, then list_library_documents() to get document ID. This function checks for existing files with the same name and validates CSV format.",
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
        validate_csv: bool = Field(
            default=True,
            description="Whether to validate the file as CSV and check for 'label' column (default: True)",
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
                # For exact same filename, check if content is identical
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_content = f.read()

                if existing_content == text_content:
                    return f"✅ File already exists with identical content at: datasets/{filename}\n📄 Source: {document_name}\n💡 Ready for training!"
                else:
                    return f"❌ File already exists with different content at: datasets/{filename}\n📄 Source: {document_name}\nUse a different custom_filename to save with a new name."

            # Validate CSV format if requested
            csv_validation_info = ""
            if validate_csv:
                try:
                    # Try to parse as CSV
                    df = pd.read_csv(io.StringIO(text_content))

                    # Check for required 'label' column
                    if "label" not in df.columns:
                        csv_validation_info = f"\n⚠️ CSV Warning: No 'label' column found. Columns: {list(df.columns)}\n💡 For ML training, add a 'label' column or use a different dataset."
                    else:
                        unique_labels = df["label"].nunique()
                        csv_validation_info = f"\n✅ CSV Valid: Found 'label' column with {unique_labels} unique classes\n📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns"

                        # Show label distribution
                        label_dist = df["label"].value_counts().to_dict()
                        csv_validation_info += f"\n📈 Label distribution: {label_dist}"

                except Exception as e:
                    csv_validation_info = (
                        f"\n⚠️ CSV Warning: Could not parse as CSV: {str(e)}"
                    )

            # Save text content to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            # Return relative path from project root
            relative_path = f"datasets/{filename}"

            return f"✅ Document saved successfully!\n📍 Dataset saved at: {relative_path}\n📄 Source: {document_name}\n📊 Size: {len(text_content)} characters{csv_validation_info}"

        except Exception as e:
            return f"Error saving document {document_id} to file: {str(e)}"

    @mcp.tool(
        title="Smart Dataset Workflow",
        description="Complete workflow: extract document from library, save as dataset, validate for ML training, and optionally start training. This is the main workflow function you should use.",
    )
    def smart_dataset_workflow(
        library_id: str = Field(
            description="ID of the library containing the document (get from list_user_libraries())"
        ),
        document_id: str = Field(
            description="ID of the document to process (get from list_library_documents())"
        ),
        dataset_name: str = Field(
            default="",
            description="Custom name for the dataset file (optional, will use document name if not provided)",
        ),
        auto_train: bool = Field(
            default=False,
            description="Whether to automatically start training after dataset is ready (default: False)",
        ),
        model_type: str = Field(
            default="random_forest",
            description="Type of model for auto-training: random_forest, svm, logistic_regression, gradient_boosting",
        ),
    ) -> str:
        """
        Complete smart workflow for dataset extraction and optional training.

        This function:
        1. Extracts document text from library
        2. Saves it as a dataset in datasets/ directory
        3. Validates CSV structure and checks for ML readiness
        4. Optionally starts training if auto_train=True

        Returns comprehensive status and next steps.
        """
        try:
            # Step 1: Save document to dataset
            save_result = save_document_text_to_file(
                library_id=library_id,
                document_id=document_id,
                custom_filename=dataset_name,
                validate_csv=True,
            )

            # Extract dataset path from save result if successful
            if "❌" in save_result:
                return save_result

            # Parse the dataset path from the success message
            import re

            path_match = re.search(r"Dataset saved at: ([^\n]+)", save_result)
            if not path_match:
                return f"❌ Could not determine dataset path from save result\n\n{save_result}"

            dataset_path = path_match.group(1)

            # Step 2: Get document name for context
            documents = mistral_client.beta.libraries.documents.list(
                library_id=library_id
            ).data
            document_name = "Unknown"
            for doc in documents:
                if doc.id == document_id:
                    document_name = doc.name
                    break

            # Step 3: Enhanced validation and analysis
            workflow_results = []
            workflow_results.append("🚀 Smart Dataset Workflow Complete!")
            workflow_results.append("=" * 50)
            workflow_results.append(f"📄 Source Document: {document_name}")
            workflow_results.append(f"📁 Dataset Location: {dataset_path}")
            workflow_results.append("")
            workflow_results.append("✅ Dataset Processing Results:")
            workflow_results.append(
                save_result.split("✅ Document saved successfully!", 1)[1]
                if "✅ Document saved successfully!" in save_result
                else save_result
            )
            workflow_results.append("")

            # Step 4: Check if ready for training
            training_ready = (
                "✅" in save_result and "Found 'label' column" in save_result
            )

            if training_ready:
                workflow_results.append("🎯 ML Training Status: READY")

                # Auto-training if requested
                if auto_train:
                    workflow_results.append("")
                    workflow_results.append("🤖 Auto-Training Initiated...")

                    # Import the ML training function
                    from mltools.mcp_tools import ml_manager
                    from mltools.models import ModelType

                    try:
                        model_type_enum = ModelType(model_type)
                        success, user_uuid, training_message = (
                            ml_manager.train_model_from_csv_path(
                                csv_path=dataset_path, model_type=model_type_enum
                            )
                        )

                        workflow_results.append("")
                        workflow_results.append("📊 Training Results:")
                        workflow_results.append(training_message)

                        if success:
                            workflow_results.append("")
                            workflow_results.append(
                                "🎉 WORKFLOW COMPLETE: Dataset extracted, saved, and model trained!"
                            )
                            workflow_results.append(f"🔑 Your model UUID: {user_uuid}")
                        else:
                            workflow_results.append("")
                            workflow_results.append(
                                "⚠️ Training failed, but dataset is ready for manual training."
                            )

                    except ValueError:
                        workflow_results.append(
                            f"❌ Invalid model type '{model_type}'. Valid options: random_forest, svm, logistic_regression, gradient_boosting"
                        )
                        workflow_results.append(
                            "💡 Dataset is ready - use train_ml_model manually with correct model type"
                        )
                    except Exception as e:
                        workflow_results.append(f"❌ Auto-training failed: {str(e)}")
                        workflow_results.append(
                            "💡 Dataset is ready - use train_ml_model manually"
                        )
                else:
                    workflow_results.append("")
                    workflow_results.append("📋 Next Steps for Manual Training:")
                    workflow_results.append(
                        f"   • Use train_ml_model with path: {dataset_path}"
                    )
                    workflow_results.append(
                        f"   • Choose model type: random_forest, svm, logistic_regression, gradient_boosting"
                    )

            else:
                workflow_results.append("⚠️ ML Training Status: REQUIRES ATTENTION")
                workflow_results.append(
                    "💡 The dataset needs a 'label' column for supervised learning"
                )
                workflow_results.append("📋 Next Steps:")
                workflow_results.append("   • Add a 'label' column to your CSV")
                workflow_results.append(
                    "   • Or use this dataset for unsupervised learning"
                )

            return "\n".join(workflow_results)

        except Exception as e:
            return f"❌ Smart workflow failed: {str(e)}"

    @mcp.tool(
        title="Quick Train from Datasets",
        description="Quick training function that lists available datasets and trains a selected one. Perfect when you have datasets ready and want to quickly start training.",
    )
    def quick_train_from_datasets(
        dataset_filename: str = Field(
            description="Filename of the dataset in datasets/ directory (e.g., 'my_data.csv')"
        ),
        model_type: str = Field(
            default="random_forest",
            description="Type of model to train: random_forest, svm, logistic_regression, gradient_boosting",
        ),
    ) -> str:
        """
        Quick training function with dataset validation and enhanced feedback.

        This function:
        1. Checks if the dataset exists in datasets/ directory
        2. Validates it's ready for ML training
        3. Starts training with the chosen model
        4. Returns comprehensive results
        """
        try:
            # Import ML manager
            from mltools.mcp_tools import ml_manager
            from mltools.models import ModelType
            from pathlib import Path

            # Construct dataset path
            dataset_path = f"datasets/{dataset_filename}"

            # Check if file exists
            if not Path(dataset_path).exists():
                # List available datasets to help user
                available_datasets = list_dataset_files()
                return f"❌ Dataset not found: {dataset_path}\n\n📁 Available Datasets:\n{available_datasets}\n\n💡 Use exact filename from the list above."

            # Validate model type
            try:
                model_type_enum = ModelType(model_type)
            except ValueError:
                return f"❌ Invalid model type '{model_type}'. Valid options: {', '.join([t.value for t in ModelType])}"

            # Pre-training validation
            try:
                df = pd.read_csv(dataset_path)
                if "label" not in df.columns:
                    return f"❌ Dataset validation failed: No 'label' column found\n📊 Columns in {dataset_filename}: {list(df.columns)}\n💡 Add a 'label' column for supervised learning."

                validation_info = []
                validation_info.append(f"✅ Pre-training validation passed!")
                validation_info.append(f"📊 Dataset: {dataset_filename}")
                validation_info.append(
                    f"📈 Shape: {df.shape[0]} rows, {df.shape[1]} columns"
                )
                validation_info.append(
                    f"🎯 Classes: {df['label'].nunique()} ({df['label'].unique().tolist()})"
                )
                validation_info.append(f"🤖 Model: {model_type}")
                validation_info.append("")
                validation_info.append("🚀 Starting training...")

                pre_validation = "\n".join(validation_info)

            except Exception as e:
                return f"❌ Dataset validation failed: {str(e)}\n💡 Ensure {dataset_filename} is a valid CSV file."

            # Start training
            success, user_uuid, training_message = ml_manager.train_model_from_csv_path(
                csv_path=dataset_path, model_type=model_type_enum
            )

            # Combine results
            final_results = []
            final_results.append(pre_validation)
            final_results.append("")
            final_results.append(training_message)

            if success:
                final_results.append("")
                final_results.append("🎉 QUICK TRAINING COMPLETE!")
                final_results.append("📋 What you can do next:")
                final_results.append(
                    f"   • Make predictions: predict_with_model(user_uuid='{user_uuid}', csv_path='your_test_data.csv')"
                )
                final_results.append(
                    f"   • Get model details: get_model_info(user_uuid='{user_uuid}')"
                )
                final_results.append(f"   • List all models: list_trained_models()")

            return "\n".join(final_results)

        except Exception as e:
            return f"❌ Quick training failed: {str(e)}"

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
