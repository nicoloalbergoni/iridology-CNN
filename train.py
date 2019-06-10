import numpy as np
import configparser
import traceback
import Preprocessing.config as config
from Preprocessing.exceptions import ConfigurationFileNotFoundError
from ML_CNN.data_preparation import create_training_data
from ML_CNN.model import create_model, train_model
from ML_CNN.utils import check_folders


def main():
    try:
        config.load_config_file('./config.ini')
    except KeyError as e:
        print('File di configurazione non corretto: manca la sezione', e)
        return
    except ConfigurationFileNotFoundError as e:
        print(e)
        return
    except configparser.ParsingError as e:
        print(e)
        return
    except Exception as e:
        traceback.print_exc()
        return

    if check_folders('./TEMP_SEG') is False:
        raise Exception(
            'Non sono presenti immagini nelle cartelle DB_PROBS_SEG e/o DB_NORMAL_SEG')

    X = []
    y = []

    training_data = create_training_data()

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)

    model, modelname = create_model(X, y, layer_size=config.NEURAL_NETWORK_MODEL.getint('LAYER_SIZE'),
                                    dense_layer=config.NEURAL_NETWORK_MODEL.getint('DENSE_LAYER'),
                                    conv_layer=config.NEURAL_NETWORK_MODEL.getint('CONV_LAYER'))
    train_model(model, modelname, X, y, batch_size=config.NEURAL_NETWORK_TRAIN.getint('BATCH_SIZE'),
                epochs=config.NEURAL_NETWORK_TRAIN.getint('EPOCHS'),
                validation_split=config.NEURAL_NETWORK_TRAIN.getfloat('VALIDATION_SPLIT'),
                tensorboard=config.NEURAL_NETWORK_TRAIN.getboolean('TENSORBOARD'))


if __name__ == '__main__':
    main()
