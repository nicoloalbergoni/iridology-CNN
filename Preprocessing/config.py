import configparser
import os

from Preprocessing.exceptions import ConfigurationFileNotFoundError

config = configparser.ConfigParser()
PREPROCESSING = None
UTILS = None
FILTERING_PUPIL = None
FILTERING_IRIS = None
THRESHOLD_PUPIL = None
THRESHOLD_IRIS = None
HOUGH_PUPIL = None
HOUGH_IRIS = None
NEURAL_NETWORK_MODEL = None
NEURAL_NETWORK_TRAIN = None


def load_config_file(path):
    """
    Load and parse the configuration file.
    This function creates a dictionary for every section in the configuration file.

    :param path: path of the configuration file
    :type path: str
    """
    if not os.path.exists(path):
        raise ConfigurationFileNotFoundError('Configuration file not found')

    config.read(path)
    global PREPROCESSING, UTILS, FILTERING_PUPIL, FILTERING_IRIS, THRESHOLD_PUPIL, THRESHOLD_IRIS, HOUGH_PUPIL, \
        HOUGH_IRIS, NEURAL_NETWORK_MODEL, NEURAL_NETWORK_TRAIN

    PREPROCESSING = config['PREPROCESSING']
    UTILS = config['UTILS']
    FILTERING_PUPIL = config['FILTERING_PUPIL']
    FILTERING_IRIS = config['FILTERING_IRIS']
    THRESHOLD_PUPIL = config['THRESHOLD_PUPIL']
    THRESHOLD_IRIS = config['THRESHOLD_IRIS']
    HOUGH_PUPIL = config['HOUGH_PUPIL']
    HOUGH_IRIS = config['HOUGH_IRIS']
    NEURAL_NETWORK_MODEL = config['NEURAL_NETWORK_MODEL']
    NEURAL_NETWORK_TRAIN = config['NEURAL_NETWORK_TRAIN']
