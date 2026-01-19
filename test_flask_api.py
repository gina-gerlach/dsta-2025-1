"""
Example client script for testing the Flask REST API.

This script demonstrates how to:
1. Load a sample from the MNIST dataset
2. Encode the image as base64
3. Send it to the Flask REST API
4. Display the prediction result

Usage:
    1. Start the Flask server: docker-compose up
    2. Run this script: python test_flask_api.py

"""
import requests
import base64
import numpy as np
from PIL import Image
import io
import sys


def load_mnist_sample(index: int = 0):
    """
    Load a sample from the MNIST dataset.
    """
    from tensorflow import keras
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_test[index], y_test[index]


def image_to_base64(image_array: np.ndarray) -> str:
    """
    Convert a numpy array to base64 string.
    """
    # Ensure values are in 0-255 range
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)

    # Convert to PIL Image
    img = Image.fromarray(image_array)

    # Save to bytes buffer as PNG
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode to base64
    return base64.b64encode(buffer.read()).decode('utf-8')


def send_prediction_request(
    image_base64: str,
    true_label: int = None,
    base_url: str = "http://localhost:5000"
) -> dict:
    """
    Send a prediction request to the Flask API.
    """
    url = f"{base_url}/predict"

    payload = {"image": image_base64}
    if true_label is not None:
        payload["true_label"] = int(true_label)

    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    return response.status_code, response.json()


def check_health(base_url: str = "http://localhost:5000") -> bool:
    """
    Check if the Flask server is healthy.
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    """Run the example client."""
    base_url = "http://localhost:5000"

    print("=" * 60)
    print("Flask REST API Client - MNIST Prediction Test")
    print("=" * 60)

    # Check server health
    print("\n[1] Checking server health...")
    if not check_health(base_url):
        print("Error: Server is not responding.")
        print("Make sure the Flask server is running: docker-compose up")
        sys.exit(1)
    print("Server is healthy!")

    # Load MNIST samples and test predictions
    print("\n[2] Loading MNIST test samples...")

    # Test multiple samples
    test_indices = [0, 1, 2, 3, 4]  # First 5 test samples

    print("\n[3] Sending prediction requests...")
    print("-" * 60)

    correct_predictions = 0
    for idx in test_indices:
        image, true_label = load_mnist_sample(idx)

        # Convert to base64
        image_base64 = image_to_base64(image)

        # Send request
        status_code, result = send_prediction_request(
            image_base64,
            true_label=true_label,
            base_url=base_url
        )

        if status_code == 200:
            predicted = result['prediction']
            confidence = result['confidence']
            is_correct = predicted == true_label

            if is_correct:
                correct_predictions += 1

            print(f"Sample {idx}: True={true_label}, "
                  f"Predicted={predicted}, "
                  f"Confidence={confidence:.2%}, "
                  f"{'CORRECT' if is_correct else 'WRONG'}")
        else:
            print(f"Sample {idx}: Error - {result}")

    print("-" * 60)
    accuracy = correct_predictions / len(test_indices) * 100
    print(f"\nResults: {correct_predictions}/{len(test_indices)} correct ({accuracy:.0f}%)")

    # Show detailed result for first sample
    print("\n[4] Detailed result for first sample:")
    image, true_label = load_mnist_sample(0)
    image_base64 = image_to_base64(image)
    status_code, result = send_prediction_request(
        image_base64,
        true_label=true_label,
        base_url=base_url
    )

    if status_code == 200:
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Input Data ID: {result['input_data_id']}")
        print(f"  Prediction ID: {result['prediction_id']}")
        print(f"  Probabilities:")
        for digit, prob in enumerate(result['probabilities']):
            bar = '#' * int(prob * 50)
            print(f"    {digit}: {prob:.4f} {bar}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
