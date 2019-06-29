import os
import cv2
import random
import pickle
import numpy as np
from tqdm import tqdm


def create_training_data(savedata=False):
    """
    Creates the vectors of features and labels for the training algorithm.
    The vector of features is represented as X while the vector of labels is represented as y.

    :param savedata: if true saves the vectors X and y localli in pickles files
    :type savedata: bool
    :return: Vector of features and labels
    :rtype: Tuple[numpy.ndarray, List[int]]
    """
    SEGMENTDIR = "./TEMP_SEG"
    CATEGORIES = ['DB_NORMAL_SEG', 'DB_PROBS_SEG']
    training_data = []
    X = []
    y = []

    for category in CATEGORIES:
        path = os.path.join(SEGMENTDIR, category)
        class_num = CATEGORIES.index(category)  # 0 = Normal 1 = Problem

        for image_path in tqdm(os.listdir(path)):
            try:
                img = cv2.imread(os.path.join(path, image_path), cv2.IMREAD_GRAYSCALE)
                training_data.append([img, class_num])
            except Exception as e:
                # TODO: aggiungere catch eccezzioni per lettura
                pass

    random.shuffle(training_data)
    # TODO: Aggiungere controlli in caso di training_data empty

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)

    if savedata is True:
        SAVEDIR = './TRAIN_DATA_DUMP'

        if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)

        pickle_out = open(os.path.join(SAVEDIR, "X.pickle"), "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open(os.path.join(SAVEDIR, "y.pickle"), "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    return X, y
