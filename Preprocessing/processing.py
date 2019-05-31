import cv2
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

import Preprocessing.config as config
from Preprocessing.display import draw_ellipse
from Preprocessing.exceptions import CircleNotFoundError
from Preprocessing.filtering import filtering, threshold


def pupil_recognition(image, thresholdpupil=70):
    f_image = filtering(image, invgray=config.FILTERING_PUPIL.getboolean('INVERT_GRAYSCALE'),
                        grayscale=config.FILTERING_PUPIL.getboolean('GRAYSCALE'),
                        sharpen=config.FILTERING_PUPIL.getboolean('SHARPEN'))
    # f_image = adjust_gamma(f_image, 2)

    thresh = threshold(f_image, thresholdpupil,
                       adaptive=config.THRESHOLD_PUPIL.getboolean('ADAPTIVE'),
                       binaryInv=config.THRESHOLD_PUPIL.getboolean('INVERTED_BINARY'),
                       otsu=config.THRESHOLD_PUPIL.getboolean('OTSU'),
                       dilate=config.THRESHOLD_PUPIL.getboolean('DILATE'))
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, config.HOUGH_PUPIL.getfloat('INVERSE_RATIO'), image.shape[0],
        param1=config.HOUGH_PUPIL.getint('PARAM1'), param2=config.HOUGH_PUPIL.getint('PARAM2'),
        minRadius=config.HOUGH_PUPIL.getint('MIN_RADIUS'), maxRadius=config.HOUGH_PUPIL.getint('MAX_RADIUS'))

    cv2.imshow('Pupil Threshold', thresh)

    if circles is None:
        # TODO: Gestire il caso in cui non trova cerchi
        raise CircleNotFoundError('Nessun cerchio trovato nella pupilla')
    elif circles.shape[0:2] == (1, 1):
        return circles[0, 0]
    else:
        # TODO: Gestire il caso in cui trovo più cerchi
        return None


def iris_recognition(image, thresholdiris=160):
    # f_image = increase_brightness(image, value=50)
    # f_image = adjust_gamma(image)
    f_image = filtering(image, invgray=config.FILTERING_IRIS.getboolean('INVERT_GRAYSCALE'),
                        grayscale=config.FILTERING_IRIS.getboolean('GRAYSCALE'),
                        sharpen=config.FILTERING_IRIS.getboolean('SHARPEN'))
    thresh = threshold(f_image, thresholdiris,
                       adaptive=config.THRESHOLD_IRIS.getboolean('ADAPTIVE'),
                       binaryInv=config.THRESHOLD_IRIS.getboolean('INVERTED_BINARY'),
                       otsu=config.THRESHOLD_IRIS.getboolean('OTSU'), dilate=config.THRESHOLD_IRIS.getboolean('DILATE'))

    canny = cv2.Canny(thresh, config.HOUGH_IRIS.getint('CANNY_TH1'), config.HOUGH_IRIS.getint('CANNY_TH2'))
    circles = cv2.HoughCircles(
        canny, cv2.HOUGH_GRADIENT, config.HOUGH_IRIS.getfloat('INVERSE_RATIO'), image.shape[0],
        param1=config.HOUGH_IRIS.getint('PARAM1'), param2=config.HOUGH_IRIS.getint('PARAM2'),
        minRadius=config.HOUGH_IRIS.getint('MIN_RADIUS'), maxRadius=config.HOUGH_IRIS.getint('MAX_RADIUS'))

    cv2.imshow('Filtered', f_image)
    cv2.imshow('Iris Threshold', thresh)
    cv2.imshow('Canny', canny)

    if circles is None:
        # TODO: Gestire il caso in cui non trova cerchi
        raise CircleNotFoundError('Nessun cerchio trovato nell\'iride')
    elif circles.shape[0:2] == (1, 1):
        return circles[0, 0]
    else:
        # TODO: Gestire il caso in cui trovo più cerchi
        return None


def segmentation(image, iris_circle, pupil_circle, startangle, endangle):
    segmented = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = segmented.shape
    outer_sector = np.zeros((height, width), np.uint8)
    pupil_sector = np.zeros((height, width), np.uint8)
    draw_ellipse(outer_sector, (iris_circle[0], iris_circle[1]), (
        iris_circle[2], iris_circle[2]), 0, -startangle, -endangle, 255, thickness=-1)
    cv2.circle(pupil_sector, (pupil_circle[0], pupil_circle[1]), int(
        pupil_circle[2]), 255, thickness=-1)
    mask = cv2.subtract(outer_sector, pupil_sector)
    masked_image = cv2.bitwise_and(segmented, segmented, mask=mask)

    return masked_image, mask


def daugman_normalizaiton(original_eye, circle, pupil_radius=0, startangle=0, endangle=45):
    start_angle = (360 - endangle) * np.pi / 180
    end_angle = (360 - startangle) * np.pi / 180

    iris_coordinates = (circle[0], circle[1])

    x = int(iris_coordinates[0])
    y = int(iris_coordinates[1])

    w = int(round(circle[2]) + 0)
    h = int(round(circle[2]) + 0)

    # cv2.circle(original_eye, iris_coordinates, int(circle[2]), (255,0,0), thickness=2)
    iris_image = original_eye[y - h:y + h, x - w:x + w]

    iris_image_to_show = cv2.resize(
        iris_image, (iris_image.shape[1] * 2, iris_image.shape[0] * 2))

    q = np.arange(start_angle, end_angle, 0.01)  # theta
    inn = np.arange(int(pupil_radius), int(
        iris_image_to_show.shape[0] / 2), 1)  # radius

    cartisian_image = np.empty(
        shape=[inn.size, int(iris_image_to_show.shape[1])])
    m = interp1d([np.pi * 2, 0], [pupil_radius, iris_image_to_show.shape[1]])

    for r in tqdm(inn):
        for t in tqdm(q):
            polarX = int((r * np.cos(t)) + iris_image_to_show.shape[1] / 2)
            polarY = int((r * np.sin(t)) + iris_image_to_show.shape[0] / 2)
            cartisian_image[r][int(
                m(t) - 1)] = iris_image_to_show[polarY][polarX]

    cartisian_image = cartisian_image.astype('uint8')
    return cartisian_image