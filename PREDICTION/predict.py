import os
import configparser
import traceback
import tensorflow as tf
import numpy as np
import cv2
import random
import Preprocessing.config as config
from Preprocessing.exceptions import ConfigurationFileNotFoundError, CannotLoadImagesError
from Preprocessing.utils import save_segments, resize_segments
from preprocess import create_data


def check_folders(datadir):
    file_count = 0

    if not os.path.exists(datadir):
        os.makedirs(datadir)
    else:
        file_count = len(
            [name for name in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, name))])

    if file_count != 0:
        return True
    else:
        return False


def main():
    try:
        config.load_config_file('../config.ini')
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

    DATADIR = './DATA_TO_PREDICT'

    if check_folders(DATADIR) is False:
        raise Exception('Non sono presenti immagini nella cartella DATA_TO_PREDICT')

    try:
        cropped_images = create_data(DATADIR)
    except CannotLoadImagesError as e:
        print(e)
        return

    model = None
    for file in os.listdir('./'):
        title = file.title().lower()
        if title.split('.')[-1] == 'model':
            model = tf.keras.models.load_model(title)
            break

    if model is None:
        print('Modello non trovato')
        return

    shape = model.input_shape
    resized_segments = resize_segments(
        cropped_images, shape[1:3])

    resized_segments = np.array(resized_segments)
    resized_segments = resized_segments.reshape(-1, resized_segments.shape[1], resized_segments.shape[2], 1)
    # for i in resized_segments:
    #     cv2.imwrite('./TEMP_PREDICTION_SEGMENT/' + str(random.randint(1, 100)) + '.jpg', i)
    #     cv2.imshow('d', i)
    #     cv2.waitKey(0)

    predictions = model.predict(resized_segments)
    CATEGORIES = ['PROBS', 'NORMAL']

    for prediction in predictions[0:]:
        print(CATEGORIES[int(prediction[0])])

if __name__ == '__main__':
    main()

