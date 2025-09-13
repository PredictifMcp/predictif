"""
MCP tools for ML training and deployment on HF Spaces
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .trainer import MLManager
from .models import MLConfig, ModelType

# Global manager instance
ml_manager = MLManager()


def register_ml_tools(mcp: FastMCP):
    """Register ML tools with the MCP server"""

    @mcp.tool(
        title="Train ML Model",
        description="Train a machine learning classifier on HF Spaces and auto-deploy inference Space",
    )
    def train_ml_model(
        hf_token: str = Field(description="Hugging Face API token"),
        file_url: str = Field(description="URL of the CSV dataset file attachment from chat with features and 'label' column"),
        model_type: str = Field(default="random_forest", description="Type of model to train: random_forest, svm, logistic_regression, gradient_boosting")
    ) -> str:
        """Train ML model and deploy inference Space"""

        # Validate model type
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            return f"Invalid model type '{model_type}'. Valid options: {', '.join([t.value for t in ModelType])}"

        # Generate model name from job ID
        import uuid
        job_id = str(uuid.uuid4())
        model_name = f"predictif-{model_type.replace('_', '')}-{job_id[:8]}"

        # Create configuration
        config = MLConfig(
            token=hf_token,
            csv_content=file_url,  # Will be downloaded in training script
            model_name=model_name,
            model_type=model_type_enum
        )

        # Create and start job with pre-generated ID
        from .models import TrainingJob
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            status="pending",
            config=config
        )

        ml_manager.active_jobs[job_id] = job
        success = ml_manager.start_training(job_id)

        if success:
            return f"""Training job started successfully!

Job ID: {job_id}
Model Name: {model_name}
Model Type: {model_type_enum.value}
Training Space: {job.training_space_url}

The training will:
1. Download and process your CSV file
2. Train a {model_type_enum.value} classifier
3. Evaluate the model performance
4. Automatically deploy an inference Space with Gradio UI
5. Make the model available for predictions

Check the training Space for progress and logs."""
        else:
            return f"Failed to start training job {job_id}"

    @mcp.tool(
        title="Get Training Status",
        description="Get the status of a training job",
    )
    def get_training_status(
        job_id: str = Field(description="ID of the training job to check")
    ) -> str:
        """Get training job status"""
        job = ml_manager.get_job_status(job_id)

        if job is None:
            return f"Job {job_id} not found"

        status_info = f"""Job {job_id} Status: {job.status}

Model Name: {job.model_name}
Training Space: {job.training_space_url}"""

        if job.inference_space_url:
            status_info += f"\nInference Space: {job.inference_space_url}"

        if job.accuracy:
            status_info += f"\nAccuracy: {job.accuracy:.4f}"

        if job.feature_names:
            status_info += f"\nFeatures: {', '.join(job.feature_names)}"

        return status_info

    @mcp.tool(
        title="Predict with Model",
        description="Make predictions using a trained model by providing a CSV file with feature values",
    )
    def predict_with_model(
        hf_token: str = Field(description="Hugging Face API token"),
        model_name: str = Field(description="Name of the trained model (e.g., predictif-model-abc123de)"),
        file_url: str = Field(description="URL of the CSV file attachment from chat with feature values (same columns as training, no 'label' column)")
    ) -> str:
        """Make batch predictions with trained model"""

        try:
            # Import requests for API calls
            import requests
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            user_info = api.whoami()
            username = user_info["name"]

            # Construct inference Space URL (matching deployment format)
            inference_space_name = f"{model_name}-inference"
            inference_space_url = f"https://{username}-{inference_space_name}.hf.space"

            # Call inference API with file URL
            response = requests.post(
                f"{inference_space_url}/api/predict_batch",
                json={"file_url": file_url}
            )

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])

                if not predictions:
                    return "No predictions returned"

                # Format results
                result_lines = ["Prediction Results:"]
                for i, pred in enumerate(predictions):
                    predicted_class = pred.get("prediction", "Unknown")
                    confidence = max(pred.get("probabilities", {}).values()) if pred.get("probabilities") else 0.0
                    result_lines.append(f"Row {i+1}: Class {predicted_class} (confidence: {confidence:.4f})")

                result_lines.append(f"\nTotal predictions: {len(predictions)}")
                result_lines.append(f"Model: {model_name}")
                result_lines.append(f"Inference Space: {inference_space_url}")

                return "\n".join(result_lines)

            else:
                return f"Error calling inference API: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error making predictions: {e}"

    @mcp.tool(
        title="List Trained Models",
        description="List all trained and deployed models",
    )
    def list_trained_models() -> str:
        """List all models in the registry"""
        models = ml_manager.list_models()

        if not models:
            return "No trained models found."

        model_list = ["Available trained models:"]
        for model_name, job in models.items():
            status = "✅ Available" if job.status == "completed" else f"❌ {job.status}"
            accuracy = f" (Accuracy: {job.accuracy:.4f})" if job.accuracy else ""
            model_list.append(f"• {model_name} - {job.model_type}{accuracy} - {status}")

        return "\n".join(model_list)

    @mcp.tool(
        title="Get Model Info",
        description="Get detailed information about a specific trained model",
    )
    def get_model_info(
        model_name: str = Field(description="Name of the model to get info for")
    ) -> str:
        """Get detailed model information"""
        model = ml_manager.get_model(model_name)

        if not model:
            return f"Model '{model_name}' not found. Use list_trained_models to see available models."

        info = f"""Model Information: {model_name}

Type: {model.model_type}
Status: {model.status}
Job ID: {model.job_id}
Accuracy: {model.accuracy:.4f if model.accuracy else 'N/A'}

URLs:
• Training Space: {model.training_space_url}
• Inference Space: {model.inference_space_url or 'Not available'}

Features: {', '.join(model.feature_names) if model.feature_names else 'N/A'}
Model Parameters: {model.config.model_params}"""

        return info

    @mcp.tool(
        title="List Models by Type",
        description="List models filtered by type (random_forest, svm, logistic_regression, gradient_boosting)",
    )
    def list_models_by_type(
        model_type: str = Field(description="Type of models to list: random_forest, svm, logistic_regression, gradient_boosting")
    ) -> str:
        """List models by type"""
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            return f"Invalid model type '{model_type}'. Valid options: {', '.join([t.value for t in ModelType])}"

        models = ml_manager.get_models_by_type(model_type_enum)

        if not models:
            return f"No {model_type} models found."

        model_list = [f"Available {model_type} models:"]
        for model_name, job in models.items():
            status = "✅ Available" if job.status == "completed" else f"❌ {job.status}"
            accuracy = f" (Accuracy: {job.accuracy:.4f})" if job.accuracy else ""
            model_list.append(f"• {model_name}{accuracy} - {status}")

        return "\n".join(model_list)