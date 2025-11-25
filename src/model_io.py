from tensorflow import keras

def save_model(model, filepath='mnist_model.h5'):
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath='mnist_model.h5'):
    model = keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model