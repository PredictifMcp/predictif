"""
ML trainer for HF Spaces with scikit-learn
"""

import uuid
import json
from typing import Dict, Optional
from huggingface_hub import HfApi
from .models import MLConfig, TrainingJob, ModelRegistry, ModelType


class MLManager:
    """Manages ML training and deployment on HF Spaces"""

    def __init__(self):
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.model_registry = ModelRegistry()

    def create_training_job(self, config: MLConfig) -> TrainingJob:
        """Create a new training job"""
        job_id = str(uuid.uuid4())

        job = TrainingJob(
            job_id=job_id,
            model_name=config.model_name,
            status="pending",
            config=config
        )

        self.active_jobs[job_id] = job
        return job

    def start_training(self, job_id: str) -> bool:
        """Start training job by creating training Space"""
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        config = job.config

        try:
            api = HfApi(token=config.token)
            user_info = api.whoami()
            username = user_info["name"]

            # Create training Space
            training_space_name = f"predictif-training-{job_id[:8]}"
            training_repo_id = f"{username}/{training_space_name}"

            self._create_training_space(api, training_repo_id, config, job_id)

            job.status = "training"
            job.training_space_url = f"https://huggingface.co/spaces/{training_repo_id}"

            return True

        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            job.status = "failed"
            return False

    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get status of a training job"""
        return self.active_jobs.get(job_id)

    def complete_training_job(self, job_id: str, accuracy: float, feature_names: list, inference_space_url: str):
        """Mark a training job as completed and add to model registry"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "completed"
            job.accuracy = accuracy
            job.feature_names = feature_names
            job.inference_space_url = inference_space_url
            self.model_registry.add_model(job)

    def list_models(self) -> Dict[str, TrainingJob]:
        """List all trained models"""
        return self.model_registry.list_models()

    def get_model(self, model_name: str) -> Optional[TrainingJob]:
        """Get a specific model by name"""
        return self.model_registry.get_model(model_name)

    def get_models_by_type(self, model_type: ModelType) -> Dict[str, TrainingJob]:
        """Get models by type"""
        return self.model_registry.get_models_by_type(model_type)

    def _create_training_space(self, api: HfApi, space_repo_id: str, config: MLConfig, job_id: str):
        """Create the training Space with scikit-learn"""
        # Create Space
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="docker"
        )

        # Add HF_TOKEN as a secret to the space
        api.add_space_secret(
            repo_id=space_repo_id,
            key="HF_TOKEN",
            value=config.token
        )

        # Upload training script
        training_script = self._generate_training_script(config, job_id)
        api.upload_file(
            path_or_fileobj=training_script.encode(),
            path_in_repo="app.py",
            repo_id=space_repo_id,
            repo_type="space"
        )

        # Upload dataset URL config
        dataset_config = {"file_url": config.csv_content}
        api.upload_file(
            path_or_fileobj=json.dumps(dataset_config).encode(),
            path_in_repo="dataset_config.json",
            repo_id=space_repo_id,
            repo_type="space"
        )

        # Upload Dockerfile
        dockerfile = self._generate_dockerfile()
        api.upload_file(
            path_or_fileobj=dockerfile.encode(),
            path_in_repo="Dockerfile",
            repo_id=space_repo_id,
            repo_type="space"
        )

        # Upload requirements
        requirements = self._generate_requirements()
        api.upload_file(
            path_or_fileobj=requirements.encode(),
            path_in_repo="requirements.txt",
            repo_id=space_repo_id,
            repo_type="space"
        )

    def _generate_training_script(self, config: MLConfig, job_id: str) -> str:
        """Generate training script for specified model type"""
        # Generate model-specific imports and parameters
        model_imports = self._get_model_imports(config.model_type)
        model_creation = self._get_model_creation_code(config)

        return f"""
import pandas as pd
import numpy as np
{model_imports}
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from huggingface_hub import HfApi
import os

def main():
    print("Starting {config.model_type.value} training...")

    # Load dataset from URL
    import json
    with open("dataset_config.json", "r") as f:
        dataset_config = json.load(f)

    file_url = dataset_config["file_url"]
    print(f"Downloading dataset from: {{file_url}}")

    df = pd.read_csv(file_url)
    print(f"Dataset shape: {{df.shape}}")

    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']

    print(f"Features: {{feature_columns}}")
    print(f"Classes: {{y.unique().tolist()}}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size={config.test_size}, random_state={config.random_state}, stratify=y
    )

    # Create and train model
    {model_creation}

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {{accuracy:.4f}}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "model.pkl")

    # Save metadata
    metadata = {{
        "job_id": "{job_id}",
        "model_name": "{config.model_name}",
        "model_type": "{config.model_type.value}",
        "accuracy": accuracy,
        "feature_names": feature_columns,
        "n_classes": len(y.unique()),
        "classes": y.unique().tolist(),
        "model_params": {json.dumps(config.model_params)}
    }}

    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training completed successfully!")

    # Deploy inference Space
    deploy_inference_space()

def deploy_inference_space():
    print("Deploying inference Space...")

    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]

    inference_space_name = "{config.model_name}-inference-{job_id[:8]}"
    inference_repo_id = f"{{username}}/{{inference_space_name}}"

    try:
        # Create inference Space
        api.create_repo(
            repo_id=inference_repo_id,
            repo_type="space",
            space_sdk="gradio"
        )

        # Upload inference app
        inference_app = generate_inference_app()
        api.upload_file(
            path_or_fileobj=inference_app.encode(),
            path_in_repo="app.py",
            repo_id=inference_repo_id,
            repo_type="space"
        )

        # Upload requirements for inference space
        inference_requirements = generate_inference_requirements()
        api.upload_file(
            path_or_fileobj=inference_requirements.encode(),
            path_in_repo="requirements.txt",
            repo_id=inference_repo_id,
            repo_type="space"
        )

        # Upload model and metadata
        with open("model.pkl", "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo="model.pkl",
                repo_id=inference_repo_id,
                repo_type="space"
            )

        with open("metadata.json", "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo="metadata.json",
                repo_id=inference_repo_id,
                repo_type="space"
            )

        print(f"Inference Space deployed: https://huggingface.co/spaces/{{inference_repo_id}}")

    except Exception as e:
        print(f"Failed to deploy inference Space: {{e}}")

def generate_inference_requirements():
    return '''gradio
joblib
pandas
scikit-learn
fastapi
uvicorn'''

def generate_inference_app():
    return '''
import gradio as gr
import joblib
import json
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import threading

# Load model and metadata
model = joblib.load("model.pkl")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

feature_names = metadata["feature_names"]

def predict(*features):
    \"\"\"Make prediction with the trained model\"\"\"

    # Create input DataFrame
    input_data = pd.DataFrame([list(features)], columns=feature_names)

    # Predict
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Format results
    prob_dict = {{f"Class {{i}}": prob for i, prob in enumerate(probabilities)}}

    return f"Predicted Class: {{prediction}}", prob_dict

def predict_batch_from_url(file_url):
    \"\"\"Make batch predictions from CSV URL\"\"\"
    try:
        # Download and process CSV
        df = pd.read_csv(file_url)

        # Check if columns match
        if not all(col in df.columns for col in feature_names):
            return {{"error": f"CSV must contain columns: {{feature_names}}"}}

        # Select only the feature columns
        X = df[feature_names]

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            prob_dict = {{f"Class {{j}}": float(prob) for j, prob in enumerate(probs)}}
            results.append({{
                "prediction": int(pred),
                "probabilities": prob_dict
            }})

        return {{"predictions": results}}

    except Exception as e:
        return {{"error": str(e)}}

# FastAPI for batch predictions
app = FastAPI()

@app.post("/api/predict_batch")
async def api_predict_batch(request: dict):
    file_url = request.get("file_url")
    if not file_url:
        return JSONResponse({{"error": "file_url is required"}}, status_code=400)

    result = predict_batch_from_url(file_url)
    return JSONResponse(result)

# Gradio interface for single predictions
inputs = [gr.Number(label=name) for name in feature_names]
outputs = [
    gr.Textbox(label="Prediction"),
    gr.Label(label="Probabilities")
]

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title=f"{{metadata['model_name']}} - ML Classifier",
    description=f"Accuracy: {{metadata['accuracy']:.4f}} | Features: {{len(feature_names)}}"
)

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI in background
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Start Gradio
    interface.launch(server_port=7860)
'''

if __name__ == "__main__":
    main()
"""

    def _get_model_imports(self, model_type: ModelType) -> str:
        """Get the appropriate sklearn imports for the model type"""
        if model_type == ModelType.RANDOM_FOREST:
            return "from sklearn.ensemble import RandomForestClassifier"
        elif model_type == ModelType.SVM:
            return "from sklearn.svm import SVC"
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            return "from sklearn.linear_model import LogisticRegression"
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return "from sklearn.ensemble import GradientBoostingClassifier"
        else:
            return "from sklearn.ensemble import RandomForestClassifier"

    def _get_model_creation_code(self, config: MLConfig) -> str:
        """Generate model creation code with parameters"""
        params = []
        for key, value in config.model_params.items():
            if isinstance(value, str):
                params.append(f'{key}="{value}"')
            else:
                params.append(f'{key}={value}')

        # Add random_state to all models
        params.append(f'random_state={config.random_state}')

        param_str = ', '.join(params)

        if config.model_type == ModelType.RANDOM_FOREST:
            return f"model = RandomForestClassifier({param_str})"
        elif config.model_type == ModelType.SVM:
            return f"model = SVC({param_str}, probability=True)"  # Enable probability for predict_proba
        elif config.model_type == ModelType.LOGISTIC_REGRESSION:
            return f"model = LogisticRegression({param_str})"
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            return f"model = GradientBoostingClassifier({param_str})"
        else:
            return f"model = RandomForestClassifier({param_str})"

    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for training"""
        return """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Fix permissions for writing model files
RUN chmod 777 /app

CMD ["python", "app.py"]
"""

    def _generate_requirements(self) -> str:
        """Generate requirements.txt"""
        return """
pandas
scikit-learn
joblib
huggingface-hub
gradio
fastapi
uvicorn
"""