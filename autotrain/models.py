"""
Scikit-learn training and deployment models
"""

from typing import Dict, Optional, Literal, Any
from dataclasses import dataclass
from enum import Enum


class ModelType(str, Enum):
    """Supported ML model types"""
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"


@dataclass
class MLConfig:
    """Configuration for ML training job on HF Spaces"""
    token: str
    csv_content: str
    model_name: str  # Name for the deployed model
    model_type: ModelType = ModelType.RANDOM_FOREST

    # Common parameters
    random_state: int = 42
    test_size: float = 0.2

    # Model-specific parameters
    model_params: Dict[str, Any] = None

    # Additional parameters
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_params is None:
            # Set default parameters based on model type
            if self.model_type == ModelType.RANDOM_FOREST:
                self.model_params = {
                    "n_estimators": 100,
                    "max_depth": None
                }
            elif self.model_type == ModelType.SVM:
                self.model_params = {
                    "C": 1.0,
                    "kernel": "rbf"
                }
            elif self.model_type == ModelType.LOGISTIC_REGRESSION:
                self.model_params = {
                    "C": 1.0,
                    "max_iter": 1000
                }
            elif self.model_type == ModelType.GRADIENT_BOOSTING:
                self.model_params = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3
                }
            else:
                self.model_params = {}

        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class TrainingJob:
    """Represents a training job on HF Spaces"""
    job_id: str
    model_name: str
    status: Literal["pending", "training", "deploying", "completed", "failed"]
    config: MLConfig

    # URLs
    training_space_url: Optional[str] = None
    inference_space_url: Optional[str] = None

    # Results
    accuracy: Optional[float] = None
    feature_names: Optional[list] = None
    model_type: Optional[str] = None

    def __post_init__(self):
        if self.model_type is None:
            self.model_type = self.config.model_type.value


@dataclass
class ModelRegistry:
    """Registry for all trained models"""
    models: Dict[str, TrainingJob] = None

    def __post_init__(self):
        if self.models is None:
            self.models = {}

    def add_model(self, job: TrainingJob):
        """Add a completed training job to the model registry"""
        if job.status == "completed":
            self.models[job.model_name] = job

    def get_model(self, model_name: str) -> Optional[TrainingJob]:
        """Get a model by name"""
        return self.models.get(model_name)

    def list_models(self) -> Dict[str, TrainingJob]:
        """List all registered models"""
        return self.models.copy()

    def get_models_by_type(self, model_type: ModelType) -> Dict[str, TrainingJob]:
        """Get models filtered by type"""
        return {
            name: job for name, job in self.models.items()
            if job.model_type == model_type.value
        }