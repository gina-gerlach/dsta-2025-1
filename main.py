"""
MNIST CNN Training and Prediction Pipeline.

This script trains a convolutional neural network on the MNIST dataset,
saves the model, and demonstrates prediction on test samples.
"""
from src.train_model import train_model
from src.load_data import load_mnist_data
from src.model_io import save_model, load_model
from src.predict import predict_classes


def main():
    """Execute the complete MNIST training and prediction pipeline."""
    # Load data
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Train model
    print("\nTraining model...")
    model = train_model(epochs=5, batch_size=128)

    # Save model
    print("\nSaving model...")
    save_model(model)

    # Load model
    print("\nLoading model...")
    loaded_model = load_model()

    # Predict
    print("\nMaking predictions...")
    predictions = predict_classes(loaded_model, x_test[:10])
    actual = y_test[:10].argmax(axis=1)

    print(f"\nPredicted: {predictions}")
    print(f"Actual:    {actual}")


if __name__ == "__main__":
    main()
