import cv2
import numpy as np
from display import draw_circles, show_images
from filtering import filtering, adjust_gamma


def pupil_recognition(image, thresholdpupil=20):
    f_image = filtering(image, invgray=False)
    #c_image = adjust_gamma(f_image, 3)
    _, thresh = cv2.threshold(f_image, thresholdpupil,
                              255, cv2.THRESH_BINARY_INV)
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 0.8, image.shape[0], param1=20, param2=5, minRadius=18, maxRadius=0)

    cv2.imshow('Pupil Threshold', thresh)
    return circles


def iris_recognition(image, thresholdiris=100):
    f_image = filtering(image, invgray=False, sharpen=False)
    #c_image = adjust_gamma(f_image, 3)
    _, thresh = cv2.threshold(f_image, thresholdiris, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 0.8, image.shape[0], param1=30, param2=10, minRadius=90, maxRadius=150)

    cv2.imshow('Filtered', f_image)
    cv2.imshow('Iris Threshold', thresh)

    return circles
