"""Model creation module for MNIST CNN architecture."""
from tensorflow import keras
from tensorflow.keras import layers

from .load_data import INPUT_SHAPE, NUM_CLASSES


def create_model():
    """
    Create a convolutional neural network model for MNIST classification.

    The model architecture consists of:
    - 2 convolutional layers with ReLU activation
    - 2 max pooling layers
    - Flatten layer
    - Dropout layer (0.5)
    - Dense output layer with softmax activation

    Returns:
        keras.Sequential: Compiled Keras model
    """
    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
