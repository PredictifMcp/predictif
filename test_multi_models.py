#!/usr/bin/env python3
"""
Test script for multi-model functionality
"""

import sys
from autotrain.trainer import MLManager
from autotrain.models import MLConfig, ModelType

def test_multi_models():
    """Test multiple model types"""

    if len(sys.argv) != 3:
        print("Usage: python test_multi_models.py <hf_token> <file_url>")
        return

    hf_token = sys.argv[1]
    file_url = sys.argv[2]

    # Create manager
    ml_manager = MLManager()

    # Test different model types
    model_types = [
        ModelType.RANDOM_FOREST,
        ModelType.SVM,
        ModelType.LOGISTIC_REGRESSION,
        ModelType.GRADIENT_BOOSTING
    ]

    jobs = []

    for model_type in model_types:
        print(f"\n=== Testing {model_type.value} ===")

        # Create config for this model type
        config = MLConfig(
            token=hf_token,
            csv_content=file_url,
            model_name=f"test-{model_type.value.replace('_', '')}-123",
            model_type=model_type
        )

        print(f"Model parameters: {config.model_params}")

        # Create job
        job = ml_manager.create_training_job(config)
        jobs.append(job)
        print(f"Created job: {job.job_id}")

        # Start training
        print("Starting training...")
        success = ml_manager.start_training(job.job_id)

        if success:
            print(f"✅ {model_type.value} training started successfully!")
            print(f"Training Space: {job.training_space_url}")
        else:
            print(f"❌ {model_type.value} training failed to start")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total jobs created: {len(jobs)}")

    successful_jobs = [j for j in jobs if j.training_space_url]
    print(f"Successful starts: {len(successful_jobs)}")

    for job in successful_jobs:
        print(f"• {job.config.model_type.value}: {job.training_space_url}")

    # Test registry functions
    print(f"\n=== Testing Registry ===")

    # Simulate some completed jobs for testing
    for job in jobs[:2]:  # Mark first 2 as completed
        ml_manager.complete_training_job(
            job.job_id,
            accuracy=0.95,
            feature_names=['feature1', 'feature2'],
            inference_space_url=f"https://test-inference-{job.job_id[:8]}.hf.space"
        )

    # Test listing models
    models = ml_manager.list_models()
    print(f"Registered models: {len(models)}")
    for name, model in models.items():
        print(f"• {name}: {model.model_type} (accuracy: {model.accuracy})")

    # Test filtering by type
    rf_models = ml_manager.get_models_by_type(ModelType.RANDOM_FOREST)
    print(f"Random Forest models: {len(rf_models)}")

    return jobs

if __name__ == "__main__":
    test_multi_models()