from tensorflow import keras
from .load_data import load_mnist_data
from .create_model import create_model

def train_model(epochs=5, batch_size=128):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Create model
    model = create_model()

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest accuracy:", test_acc)

    return model
