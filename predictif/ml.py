"""
ML training and prediction business logic
"""

import uuid
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from .files import FileManager


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"


@dataclass
class TrainingJob:
    job_id: str
    model_name: str
    status: str
    model_type: str
    accuracy: Optional[float] = None
    feature_names: Optional[list] = None


class MLManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.active_jobs: Dict[str, TrainingJob] = {}
        self._load_existing_models()

    def _get_model_dir(self, model_uuid: str) -> Path:
        return self.models_dir / model_uuid

    def _load_existing_models(self):
        for user_dir in self.models_dir.iterdir():
            if user_dir.is_dir():
                model_file = user_dir / "model.pkl"
                metadata_file = user_dir / "metadata.json"

                if model_file.exists() and metadata_file.exists():
                    try:
                        model_uuid = user_dir.name
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)

                        job = TrainingJob(
                            job_id=model_uuid,
                            model_name=model_uuid,
                            status="completed",
                            model_type=metadata.get("model_type", "unknown"),
                            accuracy=metadata.get("accuracy"),
                            feature_names=metadata.get("feature_names", []),
                        )

                        self.active_jobs[model_uuid] = job
                    except Exception:
                        continue

    def _get_sklearn_model(self, model_type: ModelType, random_state: int = 42):
        model_map = {
            ModelType.RANDOM_FOREST: RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            ModelType.SVM: SVC(probability=True, random_state=random_state),
            ModelType.LOGISTIC_REGRESSION: LogisticRegression(
                random_state=random_state, max_iter=1000
            ),
            ModelType.GRADIENT_BOOSTING: GradientBoostingClassifier(
                random_state=random_state
            ),
        }
        return model_map[model_type]

    def train_model_from_file(
        self, filename: str, model_type: ModelType, test_size: float = 0.2
    ) -> Tuple[bool, str, str]:
        try:
            csv_path = Path("datasets") / filename

            # Validate test_size parameter
            if not (0.0 < test_size < 1.0):
                return False, "", f"test_size must be between 0.0 and 1.0, got {test_size}"

            # First check if dataset already exists locally
            if csv_path.exists():
                return self.train_model_from_csv_path(str(csv_path), model_type, test_size)

            # If not found locally, search in libraries and provide helpful hint
            file_manager = FileManager()
            file_info = file_manager.find_file(filename)

            if not file_info:
                return (
                    False,
                    "",
                    f"Dataset '{filename}' not found in datasets/ folder. "
                    f"File also not found in any library. "
                    f"Please ensure the file exists in your libraries, then use the 'save_document' function to download it to datasets/"
                )

            # File found in library - provide hint to agent
            return (
                False,
                "",
                f"Dataset '{filename}' not found in datasets/ folder, but found in library '{file_info['library_name']}'. "
                f"Please call save_document with library_id='{file_info['library_id']}' and document_id='{file_info['document_id']}' "
                f"to download the file before training."
            )
            if "Error" in save_result:
                return False, "", f"Failed to save file: {save_result}"

            csv_path = f"datasets/{filename}"
            return self.train_model_from_csv_path(csv_path, model_type, split_ratio)

        except Exception as e:
            return False, "", f"Training error: {str(e)}"

    def train_model_from_csv_path(
        self, csv_path: str, model_type: ModelType, test_size: float = 0.2
    ) -> Tuple[bool, str, str]:
        try:
            # Validate test_size parameter
            if not (0.0 < test_size < 1.0):
                return False, "", f"test_size must be between 0.0 and 1.0, got {test_size}"

            csv_path_obj = Path(csv_path)
            if not csv_path_obj.exists():
                return False, "", f"CSV file not found at path: {csv_path}"

            if not csv_path_obj.suffix.lower() == '.csv':
                return False, "", f"File must be a CSV file, got: {csv_path_obj.suffix}"

            df = pd.read_csv(csv_path)

            if "label" not in df.columns:
                return False, "", "Dataset must contain 'label' column"

            X = df.drop("label", axis=1)
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            model = self._get_sklearn_model(model_type)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            model_uuid = str(uuid.uuid4())
            model_dir = self._get_model_dir(model_uuid)
            model_dir.mkdir(exist_ok=True)

            joblib.dump(model, model_dir / "model.pkl")

            metadata = {
                "model_type": model_type.value,
                "accuracy": float(accuracy),
                "feature_names": list(X.columns),
                "classes": [int(cls) if hasattr(cls, 'item') else cls for cls in model.classes_],
                "n_classes": int(len(model.classes_)),
                "dataset_shape": [int(dim) for dim in df.shape],
                "test_size": float(test_size),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "trained_at": datetime.now().isoformat(),
                "model_params": model.get_params(),
            }

            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            job = TrainingJob(
                job_id=model_uuid,
                model_name=model_uuid,
                status="completed",
                model_type=model_type.value,
                accuracy=accuracy,
                feature_names=list(X.columns),
            )

            self.active_jobs[model_uuid] = job

            return (
                True,
                model_uuid,
                f"Model trained successfully with accuracy: {accuracy:.4f}",
            )

        except Exception as e:
            return False, "", f"Training failed: {str(e)}"

    def get_model(self, model_uuid: str) -> Optional[TrainingJob]:
        return self.active_jobs.get(model_uuid)

    def list_all_models(self) -> Dict[str, TrainingJob]:
        return self.active_jobs.copy()

    def get_model_info(self, model_uuid: str) -> Optional[Dict]:
        if model_uuid not in self.active_jobs:
            return None

        model_dir = self._get_model_dir(model_uuid)
        metadata_file = model_dir / "metadata.json"
        model_file = model_dir / "model.pkl"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            return {
                "model": self.active_jobs[model_uuid],
                "metadata": metadata,
                "files": {
                    "model_dir": str(model_dir),
                    "model_size_mb": round(model_file.stat().st_size / (1024 * 1024), 2)
                    if model_file.exists()
                    else 0,
                },
            }
        except Exception:
            return None

    def predict_from_csv_path(self, model_uuid: str, csv_path: str) -> Dict:
        try:
            if model_uuid not in self.active_jobs:
                return {"error": "Model not found"}

            model_dir = self._get_model_dir(model_uuid)
            model_file = model_dir / "model.pkl"

            if not model_file.exists():
                return {"error": "Model file not found"}

            model = joblib.load(model_file)
            df = pd.read_csv(csv_path)

            if "label" in df.columns:
                X = df.drop("label", axis=1)
            else:
                X = df

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                results.append(
                    {
                        "prediction": pred,
                        "probabilities": dict(zip(model.classes_, probs)),
                        "max_probability": float(max(probs)),
                    }
                )

            job = self.active_jobs[model_uuid]
            return {
                "predictions": results,
                "model_info": {
                    "model_type": job.model_type,
                    "accuracy": job.accuracy or 0.0,
                },
            }

        except Exception as e:
            return {"error": str(e)}

    def delete_model(self, model_uuid: str) -> bool:
        try:
            if model_uuid in self.active_jobs:
                model_dir = self._get_model_dir(model_uuid)
                if model_dir.exists():
                    import shutil

                    shutil.rmtree(model_dir)
                del self.active_jobs[model_uuid]
                return True
            return False
        except Exception:
            return False

