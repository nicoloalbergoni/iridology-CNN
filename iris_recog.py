import cv2
import numpy as np
from display import draw_circles, show_images
from filtering import filtering, adjust_gamma


def iris_recognition(image, thresholdiris=100):
    f_image = filtering(image, invgray=False)
    c_image = adjust_gamma(f_image)
    _, thresh = cv2.threshold(c_image, thresholdiris, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 1, image.shape[0], param1=30, param2=20, minRadius=0, maxRadius=0)

    cv2.imshow('Filtered', c_image)
    cv2.imshow('Iris Threshold', thresh)

    return circles


def pupil_recognition(image, thresholdpupil=20):
    f_image = filtering(image, invgray=False)
    c_image = adjust_gamma(f_image)
    _, thresh = cv2.threshold(c_image, thresholdpupil, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 1, image.shape[0], param1=30, param2=20, minRadius=0, maxRadius=0)

    cv2.imshow('Pupil Threshold', thresh)
    return circles
