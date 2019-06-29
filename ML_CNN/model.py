import os
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential


def create_model(X, y, layer_size=64, dense_layer=0, conv_layer=3, activation='relu', loss='binary_crossentropy',
                  optimizer='adam', conv_pool_size=3, pooling_pool_size=2):
    """
    Creates the structure of the CNN model.

    :param X: vector of features
    :type X: numpy.ndarray
    :param y: vector of labels
    :type y: List[int]
    :param layer_size: dimensionality of the output space
    :type layer_size: int
    :param dense_layer: number of dense layer to be added in sequence after the convolutional layers
    :type dense_layer: int
    :param conv_layer: number of Convolutional-MaxPooling layer to be added in sequence after the first
        convolutional layer
    :type conv_layer: int
    :param activation: type of activation function fot the hidden layers
    :type activation: str
    :param loss: type of loss function
    :type loss: str
    :param optimizer: type of optimizer
    :type optimizer: str
    :param conv_pool_size: number that specifies the size of the convolution window
    :type conv_pool_size: int
    :param pooling_pool_size: number that specifies the size of the pooling window
    :type pooling_pool_size: int
    :return: Keras object that contains the structure of the model and model name
    :rtype: Tuple[tensorflow.python.keras.engine.sequential.Sequential, str]
    """
    modelname = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}.model"

    model = Sequential()

    model.add(Conv2D(layer_size, (conv_pool_size, conv_pool_size), input_shape=X.shape[1:]))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(pooling_pool_size, pooling_pool_size)))

    for l in range(conv_layer - 1):
        model.add(Conv2D(layer_size, (conv_pool_size, conv_pool_size)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(pooling_pool_size, pooling_pool_size)))

    # converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())

    for l in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation(activation))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model, modelname


def train_model(model, modelname, X, y, batch_size=32, epochs=3, validation_split=0.3, tensorboard=True):
    """
    Trains the model based on the vectors of feature and labels X and y.
    Saves the trained model in the MODELS folder.

    :param model: model to be trained
    :type model: tensorflow.python.keras.engine.sequential.Sequential
    :param modelname: name of the model
    :type modelname: str
    :param X: array of features
    :type X: numpy.ndarray
    :param y: list of labels
    :type y: List[int]
    :param batch_size: size of a batch of data
    :type batch_size: int
    :param epochs: number of epochs
    :type epochs: int
    :param validation_split: fraction of the input dataset to be saved for validation
    :type validation_split: float
    :param tensorboard: if true enables tensorboard
    :type tensorboard: bool
    """
    MODELDIR = './MODELS'
    LOGDIR = os.path.join(MODELDIR, 'LOGS')

    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)

    std_X = X / 255.0

    if tensorboard is True:
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)

        tensorboard = TensorBoard(log_dir=os.path.join(LOGDIR, modelname))
        model.fit(std_X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=[tensorboard])
    else:
        model.fit(std_X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    model.save(os.path.join(MODELDIR, modelname))
