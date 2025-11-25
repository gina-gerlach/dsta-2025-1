import os
from tensorflow import keras

# Default directory for model persistence (Docker volume mount point)
MODEL_DIR = os.getenv('MODEL_DIR', '/app/models')
DEFAULT_MODEL_NAME = 'mnist_model.h5'

def save_model(model, filepath=None):

    if filepath is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        filepath = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)

    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath=None):

    if filepath is None:
        filepath = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)

    model = keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model