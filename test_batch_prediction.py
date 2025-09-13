#!/usr/bin/env python3
"""
Test script for batch predictions
"""

import requests
import json

def test_batch_prediction():
    """Test batch prediction functionality"""

    # URL de l'espace d'inférence (basé sur le dernier job)
    inference_space_url = "https://alexvplle-test-model-123-inference-9512225a.hf.space"
    api_url = f"{inference_space_url}/api/predict_batch"

    # URL du CSV de test (même dataset que pour l'entraînement)
    csv_url = "https://drive.google.com/uc?export=download&id=1mGPc-yk1kRAGKlMthrAevBuUAkhqvE4b"

    print("Testing batch prediction...")
    print(f"Inference Space: {inference_space_url}")
    print(f"CSV URL: {csv_url}")

    # Préparer la requête
    payload = {
        "file_url": csv_url
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # D'abord vérifier si l'espace est accessible
        print("\nChecking if inference space is accessible...")
        check_response = requests.get(inference_space_url, timeout=30)
        print(f"Space accessibility: {check_response.status_code}")

        print("\nSending batch prediction request...")
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            if "predictions" in result:
                predictions = result["predictions"]
                print(f"✅ Batch prediction successful!")
                print(f"Number of predictions: {len(predictions)}")

                # Afficher les premières prédictions
                print("\nFirst 5 predictions:")
                for i, pred in enumerate(predictions[:5]):
                    print(f"  Row {i+1}: Predicted class {pred['prediction']} with probabilities {pred['probabilities']}")

                # Statistiques
                predicted_classes = [pred['prediction'] for pred in predictions]
                class_counts = {}
                for cls in predicted_classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1

                print(f"\nClass distribution:")
                for cls, count in sorted(class_counts.items()):
                    print(f"  Class {cls}: {count} predictions")

            else:
                print(f"❌ Unexpected response format: {result}")

        else:
            print(f"❌ Request failed with status {response.status_code}")
            try:
                error_info = response.json()
                print(f"Error details: {error_info}")
            except:
                print(f"Response text: {response.text}")

    except requests.exceptions.Timeout:
        print("❌ Request timed out after 60 seconds")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - inference space might not be ready yet")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_batch_prediction()