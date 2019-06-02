import os
import cv2
import random
from tqdm import tqdm


def create_training_data():

    SEGMENTDIR = "./TEMP_SEG"
    CATEGORIES = ['DB_NORMAL_SEG', 'DB_PROBS_SEG']
    training_data = []

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
    return training_data
