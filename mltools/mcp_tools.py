"""
MCP tools for ML training and inference
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .local_trainer import LocalMLManager
from .models import ModelType

# Global manager instance
ml_manager = LocalMLManager()


def register_ml_tools(mcp: FastMCP):
    """Register ML tools with the MCP server"""

    @mcp.tool(
        title="Train ML Model",
        description="MAIN ENTRY POINT: Train a machine learning model. If no CSV path provided, guides you through the complete workflow: find libraries → extract documents → save as dataset → train model. Always use save_document_text_to_file workflow for dataset creation.",
    )
    def train_ml_model(
        csv_path: str = Field(default="", description="Path to CSV file. If empty, will guide through library→document→dataset extraction workflow"),
        model_type: str = Field(default="random_forest", description="Type of model to train: random_forest, svm, logistic_regression, gradient_boosting"),
        library_id: str = Field(default="", description="(Workflow mode) Library ID containing the document - get from list_user_libraries()"),
        document_id: str = Field(default="", description="(Workflow mode) Document ID to extract - get from list_library_documents()"),
        dataset_name: str = Field(default="", description="(Workflow mode) Custom name for dataset file (optional)")
    ) -> str:
        """MAIN ENTRY POINT: Complete ML training workflow with automatic dataset creation from libraries"""

        # WORKFLOW MODE: If csv_path is not provided, guide through dataset creation
        if not csv_path:
            # Check if library_id and document_id are provided for workflow mode
            if not library_id or not document_id:
                return """🚀 TRAIN ML MODEL - MAIN ENTRY POINT

📋 COMPLETE WORKFLOW OPTIONS:

🎯 OPTION 1 - Direct Training (if you have a dataset ready):
   → train_ml_model(csv_path="datasets/your_file.csv", model_type="random_forest")

🎯 OPTION 2 - Full Workflow (extract dataset from libraries then train):

   STEP 1: Find your data source
   → list_user_libraries()
   This shows: [Library Name] -> ID: library_id

   STEP 2: Browse documents in the library
   → list_library_documents(library_id="YOUR_LIBRARY_ID_HERE")
   This shows: [Document Name] -> ID: document_id

   STEP 3: Train with automatic dataset extraction
   → train_ml_model(
       library_id="YOUR_LIBRARY_ID",
       document_id="YOUR_DOCUMENT_ID",
       model_type="random_forest"
     )

🔄 WORKFLOW EXPLANATION:
   • This function will automatically use save_document_text_to_file()
   • Extract document text and save to datasets/ directory
   • Validate CSV format and check for 'label' column
   • Start training immediately if dataset is ML-ready
   • Return the trained model UUID for predictions

🤖 Available model types: random_forest, svm, logistic_regression, gradient_boosting

💡 START HERE: Run list_user_libraries() to begin!"""

            # WORKFLOW MODE: Extract dataset then train
            try:
                # Import required functions from predictif.tools
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'predictif'))
                from predictif.tools import save_document_text_to_file

                workflow_results = []
                workflow_results.append("🤖 AUTOMATIC WORKFLOW INITIATED")
                workflow_results.append("=" * 50)
                workflow_results.append(f"📚 Library ID: {library_id}")
                workflow_results.append(f"📄 Document ID: {document_id}")
                workflow_results.append(f"🎯 Model Type: {model_type}")
                workflow_results.append("")
                workflow_results.append("⏳ STEP 1: Extracting document to dataset...")

                # Call save_document_text_to_file to extract dataset
                save_result = save_document_text_to_file(
                    library_id=library_id,
                    document_id=document_id,
                    custom_filename=dataset_name,
                    validate_csv=True
                )

                # Check if extraction failed
                if "❌" in save_result:
                    workflow_results.append("❌ Dataset extraction failed:")
                    workflow_results.append(save_result)
                    return "\n".join(workflow_results)

                # Parse dataset path from save result
                import re
                path_match = re.search(r"Dataset saved at: ([^\n]+)", save_result)
                if not path_match:
                    workflow_results.append("❌ Could not determine dataset path from extraction result")
                    workflow_results.append(save_result)
                    return "\n".join(workflow_results)

                extracted_dataset_path = path_match.group(1)

                workflow_results.append(f"✅ Dataset extracted successfully: {extracted_dataset_path}")
                workflow_results.append("")
                workflow_results.append("⏳ STEP 2: Starting model training...")

                # Proceed with training using the extracted dataset
                csv_path = extracted_dataset_path

            except Exception as e:
                return f"❌ Workflow mode failed: {str(e)}\n\n💡 Try direct mode with csv_path parameter or check your library_id and document_id"

        # TRAINING PHASE (both direct and workflow mode reach here)
        # Validate model type
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            return f"❌ Invalid model type '{model_type}'. Valid options: {', '.join([t.value for t in ModelType])}"

        # Validate CSV file exists
        from pathlib import Path
        if not Path(csv_path).exists():
            return f"""❌ CSV file not found: {csv_path}

🔍 DATASET TROUBLESHOOTING:

1️⃣ If you used workflow mode, the extraction may have failed
2️⃣ For direct mode, ensure the CSV file exists at the specified path
3️⃣ Use save_document_text_to_file() to create datasets from libraries
4️⃣ Expected location: datasets/your_file.csv

💡 Restart workflow with list_user_libraries() → list_library_documents() → save_document_text_to_file()"""

        # Pre-training validation
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if 'label' not in df.columns:
                return f"""❌ Dataset validation failed: No 'label' column found

📊 Current columns in {Path(csv_path).name}: {list(df.columns)}

🔧 SOLUTIONS:
1️⃣ Add a 'label' column to your CSV file for supervised learning
2️⃣ Use a different dataset that has a 'label' column
3️⃣ Extract a different document from your libraries

💡 The 'label' column should contain the target values you want to predict."""
        except Exception as e:
            return f"❌ Could not validate CSV file: {str(e)}\n💡 Ensure {csv_path} is a valid CSV file."

        # Train model from CSV path
        success, user_uuid, training_message = ml_manager.train_model_from_csv_path(
            csv_path=csv_path,
            model_type=model_type_enum
        )

        # Enhanced result formatting
        if 'workflow_results' in locals():
            # Workflow mode - combine extraction and training results
            workflow_results.append("")
            workflow_results.append("📊 TRAINING RESULTS:")
            workflow_results.append(training_message)

            if success:
                workflow_results.append("")
                workflow_results.append("🎉 COMPLETE WORKFLOW SUCCESS!")
                workflow_results.append(f"📁 Dataset: {extracted_dataset_path}")
                workflow_results.append(f"🆔 Model UUID: {user_uuid}")
                workflow_results.append("")
                workflow_results.append("🔮 NEXT STEPS:")
                workflow_results.append(f"   • Make predictions: predict_with_model(model_uuid='{user_uuid}', dataset_file='test_data.csv')")
                workflow_results.append(f"   • Get model info: get_model_info(user_uuid='{user_uuid}')")

            return "\n".join(workflow_results)
        else:
            # Direct mode - return training result
            return training_message

    @mcp.tool(
        title="Make Prediction",
        description="Make predictions using a trained model UUID with CSV data. Automatically resolves dataset paths from ./datasets/ directory.",
    )
    def predict_with_model(
        model_uuid: str = Field(description="Model UUID returned from training (e.g., from train_ml_model)"),
        dataset_file: str = Field(description="Dataset filename (e.g., 'iris.csv') - automatically looks in ./datasets/ directory")
    ) -> str:
        """Make predictions with trained model using Model UUID and automatic dataset path resolution"""

        # Enhanced validation with automatic path resolution
        from pathlib import Path

        # Automatically resolve dataset path
        if not dataset_file.startswith('./datasets/') and not dataset_file.startswith('datasets/'):
            # Auto-resolve: if user says "iris.csv", make it "./datasets/iris.csv"
            csv_path = f"./datasets/{dataset_file}"
        else:
            csv_path = dataset_file

        # 1. Validate model exists using model_uuid
        model = ml_manager.get_model(model_uuid)
        if not model:
            available_models = ml_manager.list_all_models()
            if available_models:
                models_list = "\n".join([f"   • {uuid}: {job.model_type}" for uuid, job in available_models.items()])
                return f"❌ Model not found: {model_uuid}\n\n📊 Available models:\n{models_list}\n\n💡 Use the exact UUID returned from training."
            else:
                return f"❌ Model not found: {model_uuid}\n\n📁 No trained models available.\n💡 Train a model first using train_ml_model."

        # 2. Validate CSV file exists with auto-path resolution
        csv_file = Path(csv_path)
        if not csv_file.exists():
            datasets_dir = Path("datasets")
            if datasets_dir.exists():
                available_files = [f.name for f in datasets_dir.glob("*.csv")]
                if available_files:
                    return f"❌ Dataset not found: {csv_path}\n\n💡 Available datasets:\n" + "\n".join([f"   • {f}" for f in available_files]) + "\n\n🔍 Just provide the filename (e.g., 'iris.csv') - path resolution is automatic!"
                else:
                    return f"❌ Dataset not found: {csv_path}\n\n📁 No CSV files in datasets/ directory."
            else:
                return f"❌ Dataset not found: {csv_path}\n\n📁 datasets/ directory doesn't exist."

        # 3. Get model info for feature validation using model_uuid
        model_info_detailed = ml_manager.get_model_info(model_uuid)
        if model_info_detailed:
            expected_features = model_info_detailed['metadata']['feature_names']
            model_accuracy = model_info_detailed['metadata']['accuracy']
            model_type = model_info_detailed['metadata']['model_type']

            # Pre-validate input file structure
            try:
                import pandas as pd
                input_df = pd.read_csv(csv_path)
                input_features = list(input_df.columns)

                # Check if input has label column (warn but don't fail)
                has_label = 'label' in input_features
                if has_label:
                    input_features.remove('label')
                    feature_note = f"ℹ️ Input file contains 'label' column - will be ignored for prediction."
                else:
                    feature_note = f"✅ Input file ready for prediction (no label column found)."

                # Check feature compatibility
                missing_features = set(expected_features) - set(input_features)
                extra_features = set(input_features) - set(expected_features)

                compatibility_info = []
                compatibility_info.append(f"📋 Model expects {len(expected_features)} features: {expected_features}")
                compatibility_info.append(f"📊 Input provides {len(input_features)} features: {input_features}")

                if missing_features:
                    return f"❌ Feature mismatch: Missing required features: {list(missing_features)}\n\n" + "\n".join(compatibility_info)

                if extra_features:
                    compatibility_info.append(f"⚠️ Extra features will be ignored: {list(extra_features)}")

                compatibility_info.append(feature_note)
                compatibility_info.append(f"🎯 Model accuracy: {model_accuracy:.4f}")
                compatibility_info.append(f"🤖 Model type: {model_type}")

                validation_summary = "\n🔍 Pre-prediction validation:\n" + "\n".join(compatibility_info) + "\n"

            except Exception as e:
                return f"❌ Input validation failed: {str(e)}\n💡 Ensure the file is a valid CSV."
        else:
            validation_summary = f"⚠️ Could not load model details for validation.\n"

        try:
            result = ml_manager.predict_from_csv_path(model_uuid, csv_path)

            if "error" in result:
                return f"❌ Prediction failed: {result['error']}"

            predictions = result["predictions"]
            model_info = result["model_info"]

            if not predictions:
                return "No predictions generated"

            # Format results with clearer workflow information
            result_lines = [f"🎯 PREDICTION RESULTS"]
            result_lines.append("=" * 50)
            result_lines.append(f"📊 Model UUID: {model_uuid}")
            result_lines.append(f"🤖 Model Type: {model_info['model_type']}")
            result_lines.append(f"🎯 Model Accuracy: {model_info['accuracy']:.4f}")
            result_lines.append(f"📁 Dataset: {csv_path}")
            result_lines.append(f"📈 Total predictions: {len(predictions)}")
            result_lines.append("")
            result_lines.append("🔮 Sample predictions:")

            # Show first 5 predictions
            for i, pred in enumerate(predictions[:5]):
                predicted_class = pred["prediction"]
                confidence = pred["max_probability"]
                result_lines.append(f"Row {i+1}: Class {predicted_class} (confidence: {confidence:.4f})")

            if len(predictions) > 5:
                result_lines.append(f"... and {len(predictions) - 5} more predictions")

            # Add class distribution
            class_counts = {}
            for pred in predictions:
                cls = pred["prediction"]
                class_counts[cls] = class_counts.get(cls, 0) + 1

            result_lines.append("")
            result_lines.append("Class distribution:")
            for cls, count in sorted(class_counts.items()):
                result_lines.append(f"  Class {cls}: {count} predictions")

            return validation_summary + "\n🚀 Prediction Results:\n" + "\n".join(result_lines)

        except Exception as e:
            return f"❌ Prediction error: {e}"

    @mcp.tool(
        title="List Trained Models",
        description="List all trained and available models",
    )
    def list_trained_models() -> str:
        """List all models in the registry"""
        models = ml_manager.list_all_models()

        if not models:
            return "No trained models found."

        model_list = ["Available trained models:"]
        for model_uuid, job in models.items():
            accuracy = f" (Accuracy: {job.accuracy:.4f})" if job.accuracy else ""
            model_list.append(f"• {model_uuid}: {job.model_type}{accuracy}")

        return "\n".join(model_list)

    @mcp.tool(
        title="Get Model Info",
        description="Get detailed information about a specific trained model",
    )
    def get_model_info(
        user_uuid: str = Field(description="User UUID of the model to get info for")
    ) -> str:
        """Get detailed model information"""
        model_info = ml_manager.get_model_info(user_uuid)

        if not model_info:
            return f"Model '{user_uuid}' not found. Use list_trained_models to see available models."

        model = model_info["model"]
        metadata = model_info["metadata"]
        files = model_info["files"]

        info = f"""Model Information: {user_uuid}

Type: {model.model_type}
Status: {model.status}
Accuracy: {metadata['accuracy']:.4f}
Trained: {metadata['trained_at'][:19].replace('T', ' ')}

Dataset Info:
• Shape: {metadata['dataset_shape']}
• Features: {', '.join(metadata['feature_names'])}
• Classes: {metadata['n_classes']} ({metadata['classes']})

Model Parameters: {metadata['model_params']}

Files:
• Directory: {files['model_dir']}
• Size: {files['model_size_mb']} MB"""

        return info

    @mcp.tool(
        title="Delete Model",
        description="Delete a trained model and its files",
    )
    def delete_model(
        user_uuid: str = Field(description="User UUID of the model to delete")
    ) -> str:
        """Delete a trained model"""
        success = ml_manager.delete_model(user_uuid)

        if success:
            return f"✅ Model {user_uuid} deleted successfully"
        else:
            return f"❌ Failed to delete model {user_uuid} (model not found)"