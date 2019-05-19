import os
import cv2
import random
from processing import iris_recognition, pupil_recognition, segmentation
from display import draw_circles, show_images

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


def main(path):
    images = load_image(path, extention='jpg', resize=False)
    for img in images:
        pupil_circles = pupil_recognition(img, thresholdpupil=70)
        iris_circles = iris_recognition(img, thresholdiris=160)
        masked_image = segmentation(img, iris_circles, pupil_circles, -20, 20)
        draw_circles(img, pupil_circles, iris_circles)
        cv2.imshow('Masked - img', masked_image)
        show_images(img)


main('./CASIA_DB')
