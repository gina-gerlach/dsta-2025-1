"""Model training module for MNIST dataset."""
from .load_data import load_mnist_data
from .create_model import create_model
from tensorflow.keras.optimizers import Adam, SGD

def train_model(
    epochs=5,
    batch_size=128,
    optimizer='adam',
    learning_rate=0.001,
    extra_layer=False,
    dropout_rate=0.5,
    callbacks=None
):
    """
    Train a CNN model on the MNIST dataset with configurable hyperparameters.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer (str): Optimizer to use ('adam' or 'sgd').
        learning_rate (float): Learning rate for the optimizer.
        extra_layer (bool): Whether to add an extra dense layer in the model.
        dropout_rate (float): Dropout rate for the Dropout layer.
        callbacks (list): List of Keras callbacks.

    Returns:
        keras.Model: Trained Keras model.
    """

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Pass optimizer as string and learning rate to create_model
    model = create_model(
        extra_layer=extra_layer,
        optimizer=optimizer,
        dropout_rate=dropout_rate
    )

    # Override optimizer learning rate if needed
    if optimizer.lower() == 'adam':
        model.optimizer.learning_rate = learning_rate
    elif optimizer.lower() == 'sgd':
        model.optimizer.learning_rate = learning_rate

    # Train the model
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    return model
