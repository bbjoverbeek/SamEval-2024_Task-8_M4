from keras import Sequential
from keras.src.layers import Dense, Activation, Dropout


def nn_model(model_number: int, classes: int) -> Sequential:
    """Returns a neural network model."""
    match model_number:
        case 1:
            return Sequential([
                Dense(64, activation="softmax"),
                Dropout(0.5),
                Dense(256, activation="softmax"),
                Dropout(0.5),
                Dense(32, activation="softmax"),
                Dropout(0.5),
                Dense(classes, activation="sigmoid"),
            ])
        case 2:
            return Sequential([
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(classes, activation="relu"),
            ])
