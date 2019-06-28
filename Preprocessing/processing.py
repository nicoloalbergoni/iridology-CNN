import cv2
import numpy as np
import Preprocessing.config as config
from Preprocessing.display import draw_ellipse
from Preprocessing.exceptions import CircleNotFoundError, MultipleCirclesFoundError
from Preprocessing.filtering import filtering, threshold, adjust_gamma, increase_brightness


def pupil_recognition(image, thresholdpupil=70, incBright=False, adjGamma=False):
    """
    Performs the recognition of the pupil from a given image.
    The recognition is done by using the Hough Circle Transform method.
    To get better result the function apply first the filtering function and then the threshold function

    :param image: image of the eye
    :type image: numpy.ndarray
    :param thresholdpupil: threshold value to pass to the threshold function
    :type thresholdpupil: int
    :param incBright: threshold value to pass to the threshold function
    :type incBright: bool
    :param adjGamma: if true calls the adjust_gamma function on the filtered image
    :type adjGamma: bool
    :return: an array that contains the center and the radius of the circle that identify the pupil
    :rtype: numpy.ndarray
    """
    f_image = filtering(image, invgray=config.FILTERING_PUPIL.getboolean('INVERT_GRAYSCALE'),
                        grayscale=config.FILTERING_PUPIL.getboolean(
                            'GRAYSCALE'),
                        sharpen=config.FILTERING_PUPIL.getboolean('SHARPEN'))

    if incBright is True:
        f_image = cv2.cvtColor(f_image, cv2.COLOR_GRAY2BGR)
        f_image = increase_brightness(
            f_image, value=config.FILTERING_PUPIL.getint('BRIGHTNESS_VALUE'))
        f_image = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)
    if adjGamma is True:
        f_image = cv2.cvtColor(f_image, cv2.COLOR_GRAY2BGR)
        f_image = adjust_gamma(
            f_image, config.FILTERING_PUPIL.getfloat('GAMMA_VALUE'))
        f_image = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)

    thresh = threshold(f_image, thresholdpupil,
                       adaptive=config.THRESHOLD_PUPIL.getboolean('ADAPTIVE'),
                       binaryInv=config.THRESHOLD_PUPIL.getboolean(
                           'INVERTED_BINARY'),
                       otsu=config.THRESHOLD_PUPIL.getboolean('OTSU'),
                       dilate=config.THRESHOLD_PUPIL.getboolean('DILATE'))
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, config.HOUGH_PUPIL.getfloat(
            'INVERSE_RATIO'), image.shape[0],
        param1=config.HOUGH_PUPIL.getint('PARAM1'), param2=config.HOUGH_PUPIL.getint('PARAM2'),
        minRadius=config.HOUGH_PUPIL.getint('MIN_RADIUS'), maxRadius=config.HOUGH_PUPIL.getint('MAX_RADIUS'))

    cv2.imshow('Pupil Threshold', thresh)

    if circles is None:
        # TODO: Gestire il caso in cui non trova cerchi
        raise CircleNotFoundError('No pupil has been recognized')
    elif circles.shape[0:2] == (1, 1):
        return circles[0, 0]
    else:
        # TODO: Gestire il caso in cui trovo più cerchi
        raise MultipleCirclesFoundError('Error in recognizing the pupil, more than one circle has been detected')


def iris_recognition(image, thresholdiris=160, incBright=False, adjGamma=False):
    """

    Performs the recognition of the iris from a given image.
    The recognition is done by using the Hough Circle Transform method.
    To get better result the function apply first the filtering function, then the threshold function and finally
    a Canny Edge Detector.

    :param image: image of the eye
    :type image: numpy.ndarray
    :param thresholdiris: threshold value to pass to the threshold function
    :type thresholdiris: int
    :param incBright: threshold value to pass to the threshold function
    :type incBright: bool
    :param adjGamma: if true calls the adjust_gamma function on the filtered image
    :type adjGamma: bool
    :return:  an array that contains the center and the radius of the circle that identify the iris
    :rtype: numpy.ndarray
    """
    f_image = filtering(image, invgray=config.FILTERING_IRIS.getboolean('INVERT_GRAYSCALE'),
                        grayscale=config.FILTERING_IRIS.getboolean(
                            'GRAYSCALE'),
                        sharpen=config.FILTERING_IRIS.getboolean('SHARPEN'))

    if incBright is True:
        f_image = cv2.cvtColor(f_image, cv2.COLOR_GRAY2BGR)
        f_image = increase_brightness(
            f_image, value=config.FILTERING_IRIS.getint('BRIGHTNESS_VALUE'))
        f_image = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)
    if adjGamma is True:
        f_image = cv2.cvtColor(f_image, cv2.COLOR_GRAY2BGR)
        f_image = adjust_gamma(
            f_image, config.FILTERING_IRIS.getfloat('GAMMA_VALUE'))
        f_image = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)

    thresh = threshold(f_image, thresholdiris,
                       adaptive=config.THRESHOLD_IRIS.getboolean('ADAPTIVE'),
                       binaryInv=config.THRESHOLD_IRIS.getboolean(
                           'INVERTED_BINARY'),
                       otsu=config.THRESHOLD_IRIS.getboolean('OTSU'), dilate=config.THRESHOLD_IRIS.getboolean('DILATE'))

    canny = cv2.Canny(thresh, config.HOUGH_IRIS.getint(
        'CANNY_TH1'), config.HOUGH_IRIS.getint('CANNY_TH2'))
    circles = cv2.HoughCircles(
        canny, cv2.HOUGH_GRADIENT, config.HOUGH_IRIS.getfloat(
            'INVERSE_RATIO'), image.shape[0],
        param1=config.HOUGH_IRIS.getint('PARAM1'), param2=config.HOUGH_IRIS.getint('PARAM2'),
        minRadius=config.HOUGH_IRIS.getint('MIN_RADIUS'), maxRadius=config.HOUGH_IRIS.getint('MAX_RADIUS'))

    cv2.imshow('Filtered', f_image)
    cv2.imshow('Iris Threshold', thresh)
    cv2.imshow('Canny', canny)

    if circles is None:
        # TODO: Gestire il caso in cui non trova cerchi
        raise CircleNotFoundError('No iris has been recognized')
    elif circles.shape[0:2] == (1, 1):
        return circles[0, 0]
    else:
        # TODO: Gestire il caso in cui trovo più cerchi
        raise MultipleCirclesFoundError('Error in recognizing the iris, more than one circle has been detected')


def segmentation(image, iris_circle, pupil_circle, startangle, endangle, min_radius, max_radius):
    """
    Extract a segment of the iris from the image of the eye.
    The extraction is based on two values of angles and two values of radius.

    :param image: image of the eye
    :type image: numpy.ndarray
    :param iris_circle: array that contains the center and the radius of the circle that identifies the iris
    :type iris_circle: numpy.ndarray
    :param pupil_circle: array that contains the center and the radius of the circle that identifies the pupil
    :type pupil_circle: numpy.ndarray
    :param startangle: starting angle
    :type startangle: int
    :param endangle: ending angle
    :type endangle: int
    :param min_radius: lower bound of the radius
    :type min_radius: int
    :param max_radius: upper bound of the radius
    :type max_radius: int
    :return: a black image that contains only the extracted segment as non black pixels
    :rtype: numpy.ndarray
    """
    segmented = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = segmented.shape
    outer_sector = np.zeros((height, width), np.uint8)
    inner_sector = np.zeros((height, width), np.uint8)

    if min_radius >= 100 or min_radius <= 0:
        min_radius = pupil_circle[2]
    else:
        min_radius = 0.01 * min_radius * iris_circle[2]
    if max_radius > 100 or max_radius <= 0:
        max_radius = 100
    max_radius = 0.01 * max_radius * iris_circle[2]
    if min_radius < pupil_circle[2]:
        min_radius = pupil_circle[2]

    if min_radius < max_radius:
        draw_ellipse(outer_sector, (iris_circle[0], iris_circle[1]), (
            max_radius, max_radius), 0, -startangle, -endangle, 255, thickness=-1)
        cv2.circle(inner_sector, (pupil_circle[0], pupil_circle[1]), int(
            min_radius), 255, thickness=-1)
    mask = cv2.subtract(outer_sector, inner_sector)
    masked_image = cv2.bitwise_and(segmented, segmented, mask=mask)

    cv2.imshow('Outer sector mask', outer_sector)
    cv2.imshow('Inner sector mask', inner_sector)
    cv2.imshow('Outer - Inner mask', mask)

    return masked_image
