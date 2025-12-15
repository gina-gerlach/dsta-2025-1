"""
Script to train and save the MNIST model for Milestone 3.

This script can be run locally or in a Docker container.
It will train the model and save it to the configured model directory.
"""
import os
from src.train_model import train_model
from src.model_io import save_model

def main():
    """Train and save the model to the models directory."""
    print("=" * 60)
    print("Training MNIST Model for Milestone 3")
    print("=" * 60)

    # Get model directory from environment or use default
    model_dir = os.getenv('MODEL_DIR', 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Train the model
    print("\nTraining model (this may take a few minutes)...")
    model = train_model(epochs=5, batch_size=128)

    # Save the model in Keras format
    model_path = os.path.join(model_dir, 'mnist_model.keras')
    print(f"\nSaving model to {model_path}...")
    save_model(model, filepath=model_path)

    print("\n" + "=" * 60)
    print("Model training complete!")
    print(f"The model has been saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
