import os
import cv2
import random
from iris_recog import iris_recognition, pupil_recognition
from display import draw_circles, show_images


def load_image(path, count=10):
    images = []
    images_names = []
    for file in os.listdir(path):
        title = file.title().lower()
        if title.split('.')[-1] == 'jpg':
            images_names.append(title)
            im = cv2.imread(os.path.join(path, title))
            y, x, _ = im.shape
            if y < x:
                new_y = int(y*0.1)
                new_x = int((x - (y-2*new_y)) / 2)
                #margin = int(x-new_x)
                im = im[new_y:int((y - new_y)), new_x:int(x-new_x)]
                im_r = cv2.resize(im, (300, 300))
                images.append(im_r)
            else:
                im_r = cv2.resize(im, (200, 200))
                images.append(im_r)
    return images


def main(path):
    images = load_image(path)
    for img in images:
        pupil_circles = pupil_recognition(img, thresholdpupil=32)
        iris_circles = iris_recognition(img, thresholdiris=100)
        draw_circles(img, pupil_circles, iris_circles)
        show_images(img)


main('./images')
