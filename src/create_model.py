"""Model creation module for MNIST CNN architecture."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from .load_data import INPUT_SHAPE, NUM_CLASSES

def create_model(extra_layer=False, optimizer='adam', dropout_rate=0.5):
    """
    Create a convolutional neural network model for MNIST classification.

    Args:
        extra_layer (bool): Whether to add an extra dense layer before output.
        optimizer (str or keras Optimizer): Optimizer to compile the model with.
        dropout_rate (float): Dropout rate for the Dropout layer.

    Returns:
        keras.Sequential: Compiled Keras model
    """
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
    ])

    if extra_layer:
        model.add(layers.Dense(32, activation="relu"))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    # Convert string optimizer to Keras optimizer
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        if optimizer == 'adam':
            optimizer = Adam()
        elif optimizer == 'sgd':
            optimizer = SGD()
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model
