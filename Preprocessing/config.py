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


def load_config_file(path):
    if not os.path.exists(path):
        raise ConfigurationFileNotFoundError('File di configurazione non trovato')

    config.read(path)
    global PREPROCESSING, UTILS, FILTERING_PUPIL, FILTERING_IRIS, THRESHOLD_PUPIL, THRESHOLD_IRIS, HOUGH_PUPIL, HOUGH_IRIS

    PREPROCESSING = config['PREPROCESSING']
    UTILS = config['UTILS']
    FILTERING_PUPIL = config['FILTERING_PUPIL']
    FILTERING_IRIS = config['FILTERING_IRIS']
    THRESHOLD_PUPIL = config['THRESHOLD_PUPIL']
    THRESHOLD_IRIS = config['THRESHOLD_IRIS']
    HOUGH_PUPIL = config['HOUGH_PUPIL']
    HOUGH_IRIS = config['HOUGH_IRIS']






