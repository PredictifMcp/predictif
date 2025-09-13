#!/usr/bin/env python3
"""
Test script for ML training functionality
"""

import sys
from autotrain.trainer import MLManager
from autotrain.models import MLConfig

def test_training():
    """Test the training functionality"""

    if len(sys.argv) != 3:
        print("Usage: python test_training.py <hf_token> <file_url>")
        print("Example: python test_training.py hf_xxxxx https://example.com/data.csv")
        return

    hf_token = sys.argv[1]
    file_url = sys.argv[2]

    print("Testing ML training...")
    print(f"File URL: {file_url}")

    # Create manager
    ml_manager = MLManager()

    # Create config
    config = MLConfig(
        token=hf_token,
        csv_content=file_url,  # This will be the file URL
        model_name="test-model-123"
    )

    # Create job
    job = ml_manager.create_training_job(config)
    print(f"Created job: {job.job_id}")

    # Start training
    print("Starting training...")
    success = ml_manager.start_training(job.job_id)

    if success:
        print(f"✅ Training started successfully!")
        print(f"Job ID: {job.job_id}")
        print(f"Training Space: {job.training_space_url}")
        print(f"Model Name: {job.model_name}")
    else:
        print("❌ Training failed to start")

    return job

if __name__ == "__main__":
    test_training()