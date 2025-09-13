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
        description="Train a machine learning classifier locally from a CSV file",
    )
    def train_ml_model(
        csv_path: str = Field(description="Path to the CSV dataset file with features and 'label' column"),
        model_type: str = Field(default="random_forest", description="Type of model to train: random_forest, svm, logistic_regression, gradient_boosting")
    ) -> str:
        """Train ML model locally and return user UUID for inference"""

        # Validate model type
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            return f"Invalid model type '{model_type}'. Valid options: {', '.join([t.value for t in ModelType])}"

        # Train model from CSV path
        success, user_uuid, message = ml_manager.train_model_from_csv_path(
            csv_path=csv_path,
            model_type=model_type_enum
        )

        return message

    @mcp.tool(
        title="Make Predictions",
        description="Make predictions using a trained model with CSV data",
    )
    def predict_with_model(
        user_uuid: str = Field(description="User UUID returned from training"),
        csv_path: str = Field(description="Path to CSV file with feature values (same columns as training, no 'label' column)")
    ) -> str:
        """Make predictions with trained model"""

        try:
            result = ml_manager.predict_from_csv_path(user_uuid, csv_path)

            if "error" in result:
                return f"❌ Prediction failed: {result['error']}"

            predictions = result["predictions"]
            model_info = result["model_info"]

            if not predictions:
                return "No predictions generated"

            # Format results
            result_lines = [f"✅ Predictions completed successfully!"]
            result_lines.append(f"Model: {user_uuid}")
            result_lines.append(f"Type: {model_info['model_type']}")
            result_lines.append(f"Accuracy: {model_info['accuracy']:.4f}")
            result_lines.append(f"Total predictions: {len(predictions)}")
            result_lines.append("")
            result_lines.append("Sample predictions:")

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

            return "\n".join(result_lines)

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