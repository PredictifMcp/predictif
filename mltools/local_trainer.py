"""
Local ML trainer for direct training and inference on MCP server
Models are stored as: models/{user_uuid}/model.pkl and models/{user_uuid}/metadata.json
The server generates a user_uuid that Le Chat can reuse for inference
"""

import uuid
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime

from .models import MLConfig, TrainingJob, ModelRegistry, ModelType

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class LocalMLManager:
    """Manages ML training and inference locally on the MCP server"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.model_registry = ModelRegistry()
        self._load_existing_models()

    def _get_model_dir(self, user_uuid: str) -> Path:
        """Get the directory for a model: models/{user_uuid}/"""
        return self.models_dir / user_uuid

    def _load_existing_models(self):
        """Load existing models from disk on startup"""
        # Scan for user UUID directories containing model.pkl
        for user_dir in self.models_dir.iterdir():
            if user_dir.is_dir():
                model_file = user_dir / "model.pkl"
                metadata_file = user_dir / "metadata.json"

                if model_file.exists() and metadata_file.exists():
                    try:
                        user_uuid = user_dir.name

                        with open(metadata_file) as f:
                            metadata = json.load(f)

                        # Reconstruct job from metadata
                        config = MLConfig(
                            token="",  # Not needed for local
                            csv_content="",  # Not needed for local
                            model_name=user_uuid,
                            model_type=ModelType(metadata["model_type"]),
                            model_params=metadata.get("model_params", {})
                        )

                        job = TrainingJob(
                            job_id=metadata["job_id"],
                            model_name=user_uuid,
                            status="completed",
                            config=config,
                            accuracy=metadata.get("accuracy"),
                            feature_names=metadata.get("feature_names"),
                            model_type=metadata["model_type"]
                        )

                        self.model_registry.add_model(job)
                        print(f"Loaded existing model: {user_uuid}")
                    except Exception as e:
                        print(f"Failed to load model from {user_dir}: {e}")

    def train_model_from_csv_path(self, csv_path: str, model_type: ModelType, model_params: Dict[str, Any] = None) -> Tuple[bool, str, str]:
        """
        Train a model from CSV file path with enhanced error handling and context

        Returns:
            Tuple[bool, str, str]: (success, user_uuid, message)
        """
        # Generate unique user UUID for this training session
        user_uuid = str(uuid.uuid4())

        try:
            # Enhanced loading with better error context
            from pathlib import Path
            csv_file = Path(csv_path)

            print(f"Loading dataset from: {csv_path}")

            # Provide context about file location
            if not csv_file.is_absolute():
                current_dir = Path.cwd()
                abs_path = current_dir / csv_path
                print(f"Resolved absolute path: {abs_path}")

            csv_data = pd.read_csv(csv_path)
            print(f"Successfully loaded CSV with shape: {csv_data.shape}")

            # Create training job
            job = self.create_training_job(user_uuid, model_type, model_params)

            # Train the model
            success = self.train_model(job.job_id, csv_data, user_uuid)

            if success:
                accuracy_str = f"{job.accuracy:.4f}" if job.accuracy else "N/A"

                # Enhanced success message with more context
                model_dir = self._get_model_dir(user_uuid)
                message = f"""✅ Training completed successfully!

🆔 User UUID: {user_uuid}
🤖 Model Type: {model_type.value}
📁 Dataset: {Path(csv_path).name}
🎯 Accuracy: {accuracy_str}
💾 Model saved to: {model_dir}

📋 Next Steps:
   • Use predict_with_model with UUID: {user_uuid}
   • Use get_model_info to view detailed metrics
   • Use list_trained_models to see all models

💡 Save this UUID: {user_uuid}"""
                return True, user_uuid, message
            else:
                return False, user_uuid, f"❌ Training failed for {user_uuid}. Check logs for details."

        except Exception as e:
            return False, user_uuid, f"❌ Training failed: {e}"

    def create_training_job(self, user_uuid: str, model_type: ModelType, model_params: Dict[str, Any] = None) -> TrainingJob:
        """Create a new training job"""
        job_id = str(uuid.uuid4())

        # Create config
        config = MLConfig(
            token="",  # Not needed for local training
            csv_content="",  # Will be provided directly as DataFrame
            model_name=user_uuid,
            model_type=model_type,
            model_params=model_params or {}
        )

        job = TrainingJob(
            job_id=job_id,
            model_name=user_uuid,
            status="pending",
            config=config
        )

        self.active_jobs[job_id] = job
        return job

    def train_model(self, job_id: str, csv_data: pd.DataFrame, user_uuid: str) -> bool:
        """Train a model locally"""
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        config = job.config

        try:
            job.status = "training"

            # Prepare data
            feature_columns = [col for col in csv_data.columns if col != 'label']
            X = csv_data[feature_columns]
            y = csv_data['label']

            print(f"Training {config.model_type.value} for {user_uuid}")
            print(f"Dataset shape: {csv_data.shape}")
            print(f"Features: {feature_columns}")
            print(f"Classes: {y.unique().tolist()}")

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
            )

            # Create model
            model = self._create_model(config)

            # Train
            print(f"Training model...")
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"Accuracy: {accuracy:.4f}")

            # Create model directory
            model_dir = self._get_model_dir(user_uuid)
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)

            # Save metadata
            metadata = {
                "job_id": job_id,
                "user_uuid": user_uuid,
                "model_name": user_uuid,
                "model_type": config.model_type.value,
                "accuracy": accuracy,
                "feature_names": feature_columns,
                "n_classes": len(y.unique()),
                "classes": y.unique().tolist(),
                "model_params": config.model_params,
                "trained_at": datetime.now().isoformat(),
                "test_size": config.test_size,
                "random_state": config.random_state,
                "dataset_shape": list(csv_data.shape),
                "class_distribution": y.value_counts().to_dict()
            }

            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = model_dir / "classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # Update job
            job.status = "completed"
            job.accuracy = accuracy
            job.feature_names = feature_columns

            # Add to registry
            self.model_registry.add_model(job)

            print(f"✅ Training completed! Model saved in: {model_dir}")
            return True

        except Exception as e:
            print(f"❌ Training failed: {e}")
            job.status = "failed"
            return False

    def _create_model(self, config: MLConfig):
        """Create a model instance based on config"""
        params = config.model_params.copy()
        params['random_state'] = config.random_state

        if config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(**params)
        elif config.model_type == ModelType.SVM:
            params['probability'] = True  # Enable probability for predict_proba
            return SVC(**params)
        elif config.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(**params)
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(**params)
        else:
            return RandomForestClassifier(**params)

    def predict_from_csv_path(self, user_uuid: str, csv_path: str) -> Dict[str, Any]:
        """Make predictions from CSV file path"""
        try:
            # Load CSV data from local path
            input_data = pd.read_csv(csv_path)
            return self.predict(user_uuid, input_data)
        except Exception as e:
            return {"error": f"Failed to load CSV: {e}"}

    def predict(self, user_uuid: str, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with a trained model"""
        model = self.model_registry.get_model(user_uuid)
        if not model:
            return {"error": f"Model '{user_uuid}' not found"}

        try:
            # Load model
            model_dir = self._get_model_dir(user_uuid)
            model_path = model_dir / "model.pkl"
            clf = joblib.load(model_path)

            # Load metadata to get feature names
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            feature_names = metadata["feature_names"]

            # Check if input has required features
            if not all(col in input_data.columns for col in feature_names):
                return {"error": f"Input must contain features: {feature_names}"}

            # Select only the feature columns in correct order
            X = input_data[feature_names]

            # Make predictions
            predictions = clf.predict(X)
            probabilities = clf.predict_proba(X)

            # Format results
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                prob_dict = {f"class_{j}": float(prob) for j, prob in enumerate(probs)}
                results.append({
                    "row_index": i,
                    "prediction": int(pred),
                    "probabilities": prob_dict,
                    "max_probability": float(max(probs))
                })

            return {
                "success": True,
                "predictions": results,
                "model_info": {
                    "user_uuid": user_uuid,
                    "model_type": metadata["model_type"],
                    "accuracy": metadata["accuracy"],
                    "n_predictions": len(results)
                }
            }

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

    def list_all_models(self) -> Dict[str, TrainingJob]:
        """List all trained models"""
        return self.model_registry.list_models()

    def get_model(self, user_uuid: str) -> Optional[TrainingJob]:
        """Get a specific model by user_uuid"""
        return self.model_registry.get_model(user_uuid)

    def delete_model(self, user_uuid: str) -> bool:
        """Delete a model and its files"""
        model = self.model_registry.get_model(user_uuid)
        if not model:
            return False

        try:
            # Remove model directory and all files
            model_dir = self._get_model_dir(user_uuid)
            if model_dir.exists():
                for file in model_dir.iterdir():
                    file.unlink()
                model_dir.rmdir()

            # Remove from registry
            if user_uuid in self.model_registry.models:
                del self.model_registry.models[user_uuid]

            print(f"Deleted model: {user_uuid}")
            return True

        except Exception as e:
            print(f"Failed to delete model {user_uuid}: {e}")
            return False

    def get_model_info(self, user_uuid: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information including files"""
        model = self.model_registry.get_model(user_uuid)
        if not model:
            return None

        try:
            model_dir = self._get_model_dir(user_uuid)
            metadata_path = model_dir / "metadata.json"

            if not metadata_path.exists():
                return None

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Get file sizes
            model_path = model_dir / "model.pkl"
            model_size = model_path.stat().st_size if model_path.exists() else 0

            return {
                "model": model,
                "metadata": metadata,
                "files": {
                    "model_dir": str(model_dir),
                    "model_file": "model.pkl",
                    "metadata_file": "metadata.json",
                    "model_size_bytes": model_size,
                    "model_size_mb": round(model_size / (1024 * 1024), 2),
                }
            }

        except Exception as e:
            print(f"Failed to get model info for {user_uuid}: {e}")
            return None

    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get status of a training job"""
        return self.active_jobs.get(job_id)