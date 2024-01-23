from keras import Sequential
from keras.src.layers import Dense, Activation, Dropout

NN_MODEL = Sequential(
    [
        Dense(64, activation="softmax"),
        Dropout(0.5),
        Dense(256, activation="softmax"),
        Dropout(0.5),
        Dense(32, activation="softmax"),
        Dropout(0.5),
        Dense(2, activation="sigmoid"),
    ]
)
