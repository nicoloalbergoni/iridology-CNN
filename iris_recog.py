import cv2
import numpy as np
from display import draw_circles, show_images
from filtering import filtering, adjust_gamma, threshold, increase_brightness


def pupil_recognition(image, thresholdpupil=20):
    f_image = filtering(image, invgray=False)
    #f_image = adjust_gamma(f_image, 2)
    # _, thresh = cv2.threshold(f_image, thresholdpupil,
    #                          255, cv2.THRESH_BINARY_INV)

    thresh = threshold(f_image, tValue=thresholdpupil,
                       adaptive=False, binaryInv=True)
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 0.8, image.shape[0], param1=20, param2=5, minRadius=18, maxRadius=60)

    cv2.imshow('Pupil Threshold', thresh)
    return circles


def iris_recognition(image, thresholdiris=100):
    f_image = increase_brightness(image, value=50)
    f_image = filtering(f_image, invgray=False, sharpen=False)
    # f_image = adjust_gamma(f_image, 1.5)
    thresh = threshold(f_image, tValue=thresholdiris,
                       adaptive=False, binaryInv=False, bg_area=True)
    # high_thresh, thresh = cv2.threshold(
    #     f_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh
    canny = cv2.Canny(thresh, 150, 200)
    circles = cv2.HoughCircles(
        canny, cv2.HOUGH_GRADIENT, 0.8, image.shape[0], param1=30, param2=10, minRadius=90, maxRadius=140)

    cv2.imshow('Filtered', f_image)
    cv2.imshow('Iris Threshold', thresh)
    cv2.imshow('Canny', canny)

    return circles
