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
        print('Incorrect configuration file format: missing section', e)
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
        print('No images found in the folder DB_PROBS_SEG and/or DB_NORMAL_SEG')
        return

    X, y = create_training_data(savedata=config.NEURAL_NETWORK_TRAIN.getboolean('SAVE_TRAIN_DATA'))

    model, modelname = create_model(X, y, layer_size=config.NEURAL_NETWORK_MODEL.getint('LAYER_SIZE'),
                                    dense_layer=config.NEURAL_NETWORK_MODEL.getint('DENSE_LAYER'),
                                    conv_layer=config.NEURAL_NETWORK_MODEL.getint('CONV_LAYER'),
                                    conv_pool_size=config.NEURAL_NETWORK_MODEL.getint('CONV_POOL_SIZE'),
                                    pooling_pool_size=config.NEURAL_NETWORK_MODEL.getint('POLLING_POOL_SIZE'),
                                    activation=config.NEURAL_NETWORK_MODEL.get('ACTIVATION_FUNCTION'),
                                    loss=config.NEURAL_NETWORK_MODEL.get('LOSS_FUNCTION'),
                                    optimizer=config.NEURAL_NETWORK_MODEL.get('OPTIMIZER'))

    train_model(model, modelname, X, y, batch_size=config.NEURAL_NETWORK_TRAIN.getint('BATCH_SIZE'),
                epochs=config.NEURAL_NETWORK_TRAIN.getint('EPOCHS'),
                validation_split=config.NEURAL_NETWORK_TRAIN.getfloat('VALIDATION_SPLIT'),
                tensorboard=config.NEURAL_NETWORK_TRAIN.getboolean('TENSORBOARD'))


if __name__ == '__main__':
    main()
