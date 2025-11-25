"""Prediction module for MNIST model."""


def predict(model, x):
    """
    Generate predictions for input samples.

    Args:
        model: The Keras model to use for predictions
        x: Input samples

    Returns:
        Array of predictions
    """
    predictions = model.predict(x)
    return predictions


def predict_classes(model, x):
    """
    Predict classes for input samples.

    Args:
        model: The Keras model to use for predictions
        x: Input samples

    Returns:
        Array of predicted class indices
    """
    predictions = model.predict(x)
    predicted_classes = predictions.argmax(axis=1)
    return predicted_classes