
#predictions
def predict(model, x):
    predictions = model.predict(x)
    return predictions

#predict classes
def predict_classes(model, x):
    predictions = model.predict(x)
    predicted_classes = predictions.argmax(axis=1)
    return predicted_classes