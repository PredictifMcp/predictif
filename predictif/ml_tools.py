"""
MCP tools for ML training and prediction
"""

from pathlib import Path
from pydantic import Field
from mcp.server.fastmcp import FastMCP

from .ml import MLManager, ModelType
from .files import FileManager


def register_ml_tools(mcp: FastMCP):
    ml_manager = MLManager()

    @mcp.tool(
        title="Train ML Model",
        description="Train a machine learning model using a CSV file from libraries.",
    )
    def train_ml_model(
        filename: str = Field(description="Name of the CSV file to train on"),
        model_type: str = Field(default="random_forest", description="Model type: random_forest, svm, logistic_regression, gradient_boosting"),
        split_ratio: float = Field(default=80, description="The ratio in percentage specifying how much of the data use for training and validation")
    ) -> str:
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            return f"Invalid model type '{model_type}'. Valid options: {', '.join([t.value for t in ModelType])}"

        try:
            success, user_uuid, message = ml_manager.train_model_from_file(filename, model_type_enum, split_ratio)
            if success:
                return f"Training completed!\nModel UUID: {user_uuid}\n{message}"
            else:
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
                    models_list = [f"{uuid}: {job.model_type}" for uuid, job in available_models.items()]
                    return f"Model '{model_uuid}' not found. Available models:\n" + "\n".join(models_list)
                else:
                    return f"Model '{model_uuid}' not found. No trained models available."

            file_manager = FileManager()
            file_info = file_manager.find_file(filename)
            if not file_info:
                return f"File '{filename}' not found in any library."

            save_result = file_manager.save_document(file_info['library_id'], file_info['document_id'])
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
                sample_predictions.append(f"Row {i + 1}: {pred['prediction']} (confidence: {pred['max_probability']:.3f})")

            return f"""Prediction completed!
Model: {model_info['model_type']} (accuracy: {model_info['accuracy']:.3f})
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
        title="List Supported Models",
        description="List all supported model types for training",
    )
    def list_supported_models() -> str:
        model_types = [model_type.value for model_type in ModelType]
        return f"Supported model types:\n" + "\n".join([f"• {model_type}" for model_type in model_types])

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

        return f"""Model Information: {user_uuid}

Type: {model.model_type}
Status: {model.status}
Accuracy: {metadata["accuracy"]:.4f}
Trained: {metadata["trained_at"][:19].replace("T", " ")}

Dataset Info:
• Shape: {metadata["dataset_shape"]}
• Features: {", ".join(metadata["feature_names"])}
• Classes: {metadata["n_classes"]} ({metadata["classes"]})

Files:
• Directory: {files["model_dir"]}
• Size: {files["model_size_mb"]} MB"""

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