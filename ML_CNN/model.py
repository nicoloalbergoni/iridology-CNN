import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential


def create_model(X, y, layer_size=64, dense_layer=0, conv_layer=3, activation='relu', loss='binary_crossentropy',
                  optimizer='adam', conv_pool_size= 3, pooling_pool_size= 2):

    modelname = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}.model"

    model = Sequential()

    model.add(Conv2D(layer_size, (conv_pool_size, conv_pool_size), input_shape=X.shape[1:]))
    model.add(Activation(activation))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pooling_pool_size, pooling_pool_size)))

    for l in range(conv_layer - 1):
        model.add(Conv2D(layer_size, (conv_pool_size, conv_pool_size)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(pooling_pool_size, pooling_pool_size)))

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())

    for l in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation(activation))
        # model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model, modelname


def train_model(model, modelname, X, y, batch_size=32, epochs=3, validation_split=0.3, tensorboard=True):
    MODELDIR = './MODELS'
    LOGDIR = os.path.join(MODELDIR, 'LOGS')

    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)

    # TODO: Va bene fare la normalizzazione in questo modo ?
    std_X = X / 255.0
    # std_X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)

    if tensorboard is True:
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)

        tensorboard = TensorBoard(log_dir=os.path.join(LOGDIR, modelname))
        model.fit(std_X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=[tensorboard])
    else:
        model.fit(std_X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    model.save(os.path.join(MODELDIR, modelname))
