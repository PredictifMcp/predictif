"""
MCP tools for ML training and prediction
"""

import os
from pathlib import Path
from pydantic import Field
from mcp.server.fastmcp import FastMCP

from .ml import MLManager, ModelType
from .files import FileManager


def register_ml_tools(mcp: FastMCP):
    ml_manager = MLManager()
    file_manager = FileManager()

    @mcp.tool(
        title="Train ML Model",
        description="Train a machine learning model using a CSV file from libraries.",
    )
    def train_ml_model(
        filename: str = Field(description="Name of the CSV file to train on"),
        model_type: str = Field(
            default="random_forest",
            description="Model type: random_forest, svm, logistic_regression, gradient_boosting",
        ),
        test_size: float = Field(
            default=0.2,
            description="Fraction of dataset to use for testing (0.0 < test_size < 1.0). Default is 0.2 (20% for testing)",
        ),
    ) -> str:
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            return f"Invalid model type '{model_type}'. Valid options: {', '.join([t.value for t in ModelType])}"

        try:
            success, user_uuid, message = ml_manager.train_model_from_file(
                filename, model_type_enum, test_size
            )
            if success:
                return f"Training completed!\nModel UUID: {user_uuid}\n{message}"
            else:
                # Check if this is a hint message about using save_document
                if "save_document" in message.lower():
                    return f"📁 Dataset Required: {message}"
                return f"Training failed: {message}"
        except Exception as e:
            return f"Training error: {str(e)}"

    @mcp.tool(
        title="Make Prediction",
        description="Make predictions using a trained model and dataset file from libraries.",
    )
    def predict_with_model(
        model_uuid: str = Field(description="Model UUID from training"),
        filename: str = Field(description="Name of the CSV dataset file to predict on"),
    ) -> str:
        try:
            model = ml_manager.get_model(model_uuid)
            if not model:
                available_models = ml_manager.list_all_models()
                if available_models:
                    models_list = [
                        f"{uuid}: {job.model_type}"
                        for uuid, job in available_models.items()
                    ]
                    return (
                        f"Model '{model_uuid}' not found. Available models:\n"
                        + "\n".join(models_list)
                    )
                else:
                    return (
                        f"Model '{model_uuid}' not found. No trained models available."
                    )

            file_manager = FileManager()
            file_info = file_manager.find_file(filename)
            if not file_info:
                return f"File '{filename}' not found in any library."

            save_result = file_manager.save_document(
                file_info["library_id"], file_info["document_id"]
            )
            if "Error" in save_result:
                return f"Failed to save file: {save_result}"

            csv_path = f"datasets/{filename}"
            if not Path(csv_path).exists():
                return f"File was not saved correctly at {csv_path}"

            result = ml_manager.predict_from_csv_path(model_uuid, csv_path)
            if "error" in result:
                return f"Prediction failed: {result['error']}"

            predictions = result["predictions"]
            model_info = result["model_info"]

            if not predictions:
                return "No predictions generated"

            class_counts = {}
            for pred in predictions:
                cls = pred["prediction"]
                class_counts[cls] = class_counts.get(cls, 0) + 1

            sample_predictions = []
            for i, pred in enumerate(predictions[:5]):
                sample_predictions.append(
                    f"Row {i + 1}: {pred['prediction']} (confidence: {pred['max_probability']:.3f})"
                )

            return f"""Prediction completed!
Model: {model_info["model_type"]} (accuracy: {model_info["accuracy"]:.3f})
Dataset: {csv_path}
Total predictions: {len(predictions)}

Sample predictions:
{chr(10).join(sample_predictions)}
{"..." if len(predictions) > 5 else ""}

Class distribution:
{chr(10).join([f"  {cls}: {count}" for cls, count in sorted(class_counts.items())])}"""

        except Exception as e:
            return f"Prediction error: {str(e)}"

    @mcp.tool(
        title="List Available Models Types",
        description="List all availables model types for training",
    )
    def list_supported_models() -> str:
        model_types = [model_type.value for model_type in ModelType]
        return f"Supported model types:\n" + "\n".join(
            [f"• {model_type}" for model_type in model_types]
        )

    @mcp.tool(
        title="List Trained Models",
        description="List all trained models with their detailed information and metrics",
    )
    def list_trained_models() -> str:
        trained_models = ml_manager.list_all_models()

        if not trained_models:
            return "No trained models found in the models directory."

        models_info = []
        for model_uuid, model_job in trained_models.items():
            model_info = ml_manager.get_model_info(model_uuid)

            if model_info:
                metadata = model_info["metadata"]
                files = model_info["files"]

                test_size = metadata.get("test_size", 0.2)
                train_samples = metadata.get("train_samples", "N/A")
                test_samples = metadata.get("test_samples", "N/A")

                train_ratio = f"{(1-test_size)*100:.1f}%" if isinstance(test_size, (int, float)) else "N/A"
                test_ratio = f"{test_size*100:.1f}%" if isinstance(test_size, (int, float)) else "N/A"

                model_summary = f"""┌─ Model: {model_uuid[:8]}...
├─ Type: {model_job.model_type}
├─ Status: {model_job.status}
├─ Accuracy: {metadata["accuracy"]:.4f} ({metadata["accuracy"]*100:.2f}%)
├─ Trained: {metadata["trained_at"][:19].replace("T", " ")}
├─ Dataset: {metadata["dataset_shape"]} samples, {len(metadata["feature_names"])} features
├─ Classes: {metadata["n_classes"]} ({metadata["classes"]})
├─ Split: {train_ratio} train / {test_ratio} test ({train_samples}/{test_samples})
├─ Features: {", ".join(metadata["feature_names"][:3])}{"..." if len(metadata["feature_names"]) > 3 else ""}
└─ Size: {files["model_size_mb"]} MB"""

                models_info.append(model_summary)
            else:
                models_info.append(f"┌─ Model: {model_uuid[:8]}...\n├─ Type: {model_job.model_type}\n├─ Status: {model_job.status}\n└─ Error: Unable to load metadata")

        header = f"Found {len(trained_models)} trained model{'s' if len(trained_models) != 1 else ''}:\n\n"
        return header + "\n\n".join(models_info)

    @mcp.tool(
        title="Get Model Info",
        description="Get detailed information about a trained model",
    )
    def get_model_info(
        user_uuid: str = Field(description="Model UUID to get info for"),
    ) -> str:
        model_info = ml_manager.get_model_info(user_uuid)
        if not model_info:
            return f"Model '{user_uuid}' not found."

        model = model_info["model"]
        metadata = model_info["metadata"]
        files = model_info["files"]

        test_size = metadata.get("test_size", 0.2)
        train_samples = metadata.get("train_samples", "N/A")
        test_samples = metadata.get("test_samples", "N/A")
        total_samples = train_samples + test_samples if isinstance(train_samples, int) and isinstance(test_samples, int) else "N/A"

        train_ratio = f"{(1-test_size):.1%}" if isinstance(test_size, (int, float)) else "N/A"
        test_ratio = f"{test_size:.1%}" if isinstance(test_size, (int, float)) else "N/A"

        model_params = metadata.get("model_params", {})
        key_params = []
        if model.model_type == "random_forest":
            key_params.extend([
                f"n_estimators: {model_params.get('n_estimators', 'N/A')}",
                f"max_depth: {model_params.get('max_depth', 'N/A')}",
                f"min_samples_split: {model_params.get('min_samples_split', 'N/A')}"
            ])
        elif model.model_type == "svm":
            key_params.extend([
                f"C: {model_params.get('C', 'N/A')}",
                f"kernel: {model_params.get('kernel', 'N/A')}",
                f"gamma: {model_params.get('gamma', 'N/A')}"
            ])
        elif model.model_type == "logistic_regression":
            key_params.extend([
                f"C: {model_params.get('C', 'N/A')}",
                f"max_iter: {model_params.get('max_iter', 'N/A')}",
                f"solver: {model_params.get('solver', 'N/A')}"
            ])
        elif model.model_type == "gradient_boosting":
            key_params.extend([
                f"n_estimators: {model_params.get('n_estimators', 'N/A')}",
                f"learning_rate: {model_params.get('learning_rate', 'N/A')}",
                f"max_depth: {model_params.get('max_depth', 'N/A')}"
            ])

        return f"""Model Information: {user_uuid}

Performance Metrics:
• Accuracy: {metadata["accuracy"]:.4f} ({metadata["accuracy"]*100:.2f}%)
• Model Type: {model.model_type}
• Status: {model.status}
• Trained: {metadata["trained_at"][:19].replace("T", " ")}

Dataset Configuration:
• Total Samples: {total_samples}
• Dataset Shape: {metadata["dataset_shape"]} (rows × features)
• Feature Count: {len(metadata["feature_names"])}
• Features: {", ".join(metadata["feature_names"])}
• Target Classes: {metadata["n_classes"]} classes {metadata["classes"]}

Train/Test Split Details:
• Split Ratio: {train_ratio} train / {test_ratio} test
• Test Size Parameter: {test_size}
• Training Samples: {train_samples}
• Testing Samples: {test_samples}

Model Hyperparameters:
{chr(10).join([f"• {param}" for param in key_params]) if key_params else "• Default parameters used"}

Storage Information:
• Directory: {files["model_dir"]}
• Model Size: {files["model_size_mb"]} MB
• Random State: {model_params.get('random_state', 'N/A')}"""

    @mcp.tool(
        title="Delete Model",
        description="Delete a trained model and its files",
    )
    def delete_model(
        user_uuid: str = Field(description="Model UUID to delete"),
    ) -> str:
        success = ml_manager.delete_model(user_uuid)
        if success:
            return f"Model {user_uuid} deleted successfully"
        else:
            return f"Failed to delete model {user_uuid} (model not found)"

    @mcp.tool(
        title="Upload a model",
        description="Upload a model to library"
    )
    def upload_model(
        library_name: str = Field(description='Library name to which the file should be saved'),
        model_uuid = Field(description="The UUID of the model that should be uploaded")
    ) -> str:
        library_id = file_manager.get_library_id(library_name)
        filepath = os.path.join('models', model_uuid, 'model.pkl')
        return file_manager.upload_document(library_id, filepath)
    
    @mcp.tool(
        title="Upload model metadata",
        description="Upload model metadata to library in the JSON format"
    )
    def upload_metadata(
        library_name: str = Field(description='Library name to which the file should be saved'),
        model_uuid = Field(description="The UUID of the model for which the metadata should be uploaded")
    ) -> str:
        library_id = file_manager.get_library_id(library_name)
        filepath = os.path.join('models', model_uuid, 'metadata.json')
        return file_manager.upload_document(library_id, filepath)
