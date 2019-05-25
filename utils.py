import cv2
import os
import random
import numpy as np
import tqdm

def resize_img(im, imgsize=300):
    y, x, _ = im.shape
    if y < x:
        new_y = int(y*0.1)
        new_x = int((x - (y-2*new_y)) / 2)
        #margin = int(x-new_x)
        im = im[new_y:int((y - new_y)), new_x:int(x-new_x)]
        im_r = cv2.resize(im, (imgsize, imgsize))

    else:
        im_r = cv2.resize(im, (imgsize, imgsize))

    return im_r


def load_image(path, count=10, extention='jpg', resize=True):
    images = []
    images_names = []
    for file in os.listdir(path):
        title = file.title().lower()
        if title.split('.')[-1] == extention:
            images_names.append(title)
            im = cv2.imread(os.path.join(path, title))

            im = resize_img(im) if resize else im

            images.append(im)
    random.shuffle(images)
    return images


def resize_segment(cropped_array):
    shapes = [c.shape for c in cropped_array]
    means = np.around(np.mean(shapes, axis=0)).astype(int)
    #print('Means: ', means)
    resized_segments = [cv2.resize(c, (means[1], means[0]), interpolation=cv2.INTER_AREA) for c in cropped_array]
    return resized_segments


def save_segments(resized_segments, path='./tmp_seg/'):
    if not os.path.exists(path):
        os.makedirs(path)
    for index, img in enumerate(resized_segments):
        complete_path = os.path.join(path, str(index) + '_r.jpg')
        cv2.imwrite(complete_path, img)










