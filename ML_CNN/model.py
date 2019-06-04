import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def create_model(X, y):
    X = X / 255.0
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train_model(model, X, y, batch_size=32, epochs=3, validation_split=0.3):
    MODELDIR = './MODELS'
    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)
    NAME = f'Model - {model.input_shape} - .model'
    model.save(os.path.join(MODELDIR, NAME))



