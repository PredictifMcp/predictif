#!/usr/bin/env python3
"""
Test script for local ML trainer functionality
"""

import pandas as pd
from pathlib import Path
from autotrain.local_trainer import LocalMLManager
from autotrain.models import ModelType

def create_test_dataset():
    """Create a simple test dataset"""
    # Create iris-like dataset
    data = {
        'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                        7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2,
                        6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2],
        'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
                       3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7,
                       3.3, 2.8, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6],
        'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
                        4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9,
                        6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1],
        'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
                       1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4,
                       2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5],
        'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    }

    df = pd.DataFrame(data)
    return df

def test_training():
    """Test training functionality"""
    print("=== Testing Local ML Training ===")

    # Create test dataset
    test_data = create_test_dataset()

    # Save test dataset to CSV
    test_csv_path = "test_dataset.csv"
    test_data.to_csv(test_csv_path, index=False)
    print(f"Created test dataset: {test_csv_path}")
    print(f"Dataset shape: {test_data.shape}")
    print(f"Classes: {test_data['label'].unique()}")

    # Initialize manager
    ml_manager = LocalMLManager()

    # Test different model types
    model_types = [
        ModelType.RANDOM_FOREST,
        ModelType.LOGISTIC_REGRESSION,
        ModelType.SVM
    ]

    trained_models = []

    for model_type in model_types:
        print(f"\n--- Testing {model_type.value} ---")

        # Train model
        success, user_uuid, message = ml_manager.train_model_from_csv_path(
            csv_path=test_csv_path,
            model_type=model_type
        )

        print(message)

        if success:
            trained_models.append(user_uuid)

            # Get model info
            model_info = ml_manager.get_model_info(user_uuid)
            if model_info:
                print(f"Model size: {model_info['files']['model_size_mb']} MB")
                print(f"Features: {model_info['metadata']['feature_names']}")
        else:
            print(f"❌ Training failed for {model_type.value}")

    return trained_models

def test_inference(trained_models):
    """Test inference functionality"""
    print("\n=== Testing Local ML Inference ===")

    if not trained_models:
        print("No trained models available for inference testing")
        return

    # Create test inference data (same structure as training, no label column)
    inference_data = pd.DataFrame({
        'sepal_length': [5.0, 6.0, 7.0],
        'sepal_width': [3.5, 3.0, 3.2],
        'petal_length': [1.5, 4.5, 6.0],
        'petal_width': [0.2, 1.4, 2.0]
    })

    # Save inference data to CSV
    inference_csv_path = "test_inference.csv"
    inference_data.to_csv(inference_csv_path, index=False)
    print(f"Created inference dataset: {inference_csv_path}")
    print(f"Inference data shape: {inference_data.shape}")

    # Initialize manager
    ml_manager = LocalMLManager()

    # Test inference with first trained model
    user_uuid = trained_models[0]
    print(f"\nTesting inference with model: {user_uuid}")

    # Test prediction from CSV path
    result = ml_manager.predict_from_csv_path(user_uuid, inference_csv_path)

    if "error" in result:
        print(f"❌ Inference failed: {result['error']}")
    else:
        print("✅ Inference successful!")
        print(f"Model info: {result['model_info']}")
        print(f"Number of predictions: {len(result['predictions'])}")

        # Show predictions
        for i, pred in enumerate(result['predictions'][:3]):  # Show first 3
            print(f"Row {i}: Class {pred['prediction']} (confidence: {pred['max_probability']:.3f})")

def test_model_management():
    """Test model management functionality"""
    print("\n=== Testing Model Management ===")

    ml_manager = LocalMLManager()

    # List all models
    models = ml_manager.list_all_models()
    print(f"Total models: {len(models)}")

    for model_id, job in models.items():
        print(f"• {model_id}: {job.model_type} (accuracy: {job.accuracy:.3f})")

    # Test model deletion (delete first model if exists)
    if models:
        first_model_id = list(models.keys())[0]
        print(f"\nTesting deletion of model: {first_model_id}")

        success = ml_manager.delete_model(first_model_id)
        if success:
            print("✅ Model deleted successfully")

            # Verify deletion
            remaining_models = ml_manager.list_all_models()
            print(f"Remaining models: {len(remaining_models)}")
        else:
            print("❌ Model deletion failed")

def cleanup():
    """Clean up test files"""
    print("\n=== Cleanup ===")
    test_files = ["test_dataset.csv", "test_inference.csv"]

    for file in test_files:
        path = Path(file)
        if path.exists():
            path.unlink()
            print(f"Removed {file}")

def main():
    """Run all tests"""
    print("🚀 Starting Local ML Trainer Tests")

    try:
        # Test training
        trained_models = test_training()

        # Test inference
        test_inference(trained_models)

        # Test model management
        test_model_management()

        print("\n✅ All tests completed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        cleanup()

if __name__ == "__main__":
    main()