"""
Scikit-learn training and deployment models
"""

from typing import Dict, Optional, Literal, Any
from dataclasses import dataclass


@dataclass
class MLConfig:
    """Configuration for ML training job on HF Spaces"""
    token: str
    csv_content: str
    model_name: str  # Name for the deployed model

    # RandomForest parameters (fixed)
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: int = 42

    # Additional parameters
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
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