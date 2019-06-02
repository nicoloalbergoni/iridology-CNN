import os
import random
import shutil

import cv2
import numpy as np


def resize_img(im, resize_shape):
    y, x, _ = im.shape
    if y < x:
        new_x = int((x - y) / 2)
        # margin = int(x-new_x)
        im = im[0:int(y), new_x:int(x - new_x)]
        im_r = cv2.resize(im, (resize_shape, resize_shape))
    else:
        im_r = cv2.resize(im, (resize_shape, resize_shape))
    return im_r


def load_image(path, extention='jpg', resize=True, resize_shape = 300):
    images = []
    images_names = []
    for file in os.listdir(path):
        title = file.title().lower()
        if title.split('.')[-1] == extention:
            images_names.append(title)
            im = cv2.imread(os.path.join(path, title))

            im = resize_img(im, resize_shape) if resize else im

            images.append(im)
    random.shuffle(images)
    return images


def get_average_shape(cropped_dict):
    shapes = np.concatenate(([c.shape for c in cropped_dict['DB_PROBS']], [c.shape for c in cropped_dict['DB_NORMAL']]))
    means = np.around(np.mean(shapes, axis=0)).astype(int)
    return means


def resize_segments(cropped_array, resizeshape):
    resized_segments = [cv2.resize(c, (resizeshape[1], resizeshape[0]), interpolation=cv2.INTER_AREA) for c in
                        cropped_array]
    return resized_segments


def save_segments(resized_segments, path):
    if not os.path.exists('./TEMP_SEG'):
        os.makedirs('./TEMP_SEG')
    path = os.path.join('./TEMP_SEG', path + '_SEG')
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # Se la cartella esiste cancella tutti i file al suo interno
        shutil.rmtree(path)
        os.makedirs(path)

    for index, img in enumerate(resized_segments):
        complete_path = os.path.join(path, str(index) + '.jpg')
        cv2.imwrite(complete_path, img)


def check_folders(datadir):
    file_count_normal = 0
    file_count_probs = 0
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    probs_path = os.path.join(datadir, 'DB_PROBS')
    if not os.path.exists(probs_path):
        os.makedirs(probs_path)
    else:
        file_count_probs = len(
            [name for name in os.listdir(probs_path) if os.path.isfile(os.path.join(probs_path, name))])

    normal_path = os.path.join(datadir, 'DB_NORMAL')
    if not os.path.exists(normal_path):
        os.makedirs(normal_path)
    else:
        file_count_normal = len(
            [name for name in os.listdir(normal_path) if os.path.isfile(os.path.join(normal_path, name))])

    if file_count_probs != 0 and file_count_normal != 0:
        return True
    else:
        return False


def crop_image(masked_image, offset=30, tollerance=80):
    mask = masked_image > tollerance

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = masked_image[x0 - offset: x1 + offset, y0 - offset: y1 + offset]
    # print('Cropped Image Shape', cropped.shape)

    return cropped
