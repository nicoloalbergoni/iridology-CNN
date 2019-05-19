import cv2
import numpy as np
from display import draw_ellipse
from filtering import filtering, adjust_gamma, threshold, increase_brightness


def pupil_recognition(image, thresholdpupil=20):
    f_image = filtering(image, invgray=False, grayscale=True)
    #f_image = adjust_gamma(f_image, 2)
    # _, thresh = cv2.threshold(f_image, thresholdpupil,
    #                          255, cv2.THRESH_BINARY_INV)

    thresh = threshold(f_image, tValue=thresholdpupil,
                       adaptive=False, binaryInv=True)
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 0.8, image.shape[0], param1=20, param2=5, minRadius=18, maxRadius=60)

    cv2.imshow('Pupil Threshold', thresh)

    if circles.shape[0:2] == (1, 1):
        return circles[0, 0]
    else:
        # TODO: Gestire il caso in cui trovo più cerchi
        pass


def iris_recognition(image, thresholdiris=100):
    #f_image = increase_brightness(image, value=50)
    #f_image = adjust_gamma(image)
    f_image = filtering(image, invgray=False, sharpen=False, grayscale=True)
    thresh = threshold(f_image, tValue=thresholdiris,
                       adaptive=False, binaryInv=False, otsu=False, dilate=False)
    # high_thresh, thresh = cv2.threshold(
    #     f_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh
    canny = cv2.Canny(thresh, 150, 200)
    circles = cv2.HoughCircles(
        canny, cv2.HOUGH_GRADIENT, 0.8, image.shape[0], param1=30, param2=10, minRadius=90, maxRadius=130)

    cv2.imshow('Filtered', f_image)
    cv2.imshow('Iris Threshold', thresh)
    cv2.imshow('Canny', canny)

    if circles.shape[0:2] == (1, 1):
        return circles[0, 0]
    else:
        # TODO: Gestire il caso in cui trovo più cerchi
        pass


def segmentation(image, iris_circle, pupil_circle, startangle, endangle):
    segmented = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = segmented.shape
    outer_sector = np.zeros((height, width), np.uint8)
    pupil_sector = np.zeros((height, width), np.uint8)
    draw_ellipse(outer_sector, (iris_circle[0], iris_circle[1]), (iris_circle[2], iris_circle[2]), 0, -startangle, -endangle, 255, thickness=-1)
    cv2.circle(pupil_sector, (pupil_circle[0], pupil_circle[1]), int(pupil_circle[2]), 255, thickness=-1)
    mask = cv2.subtract(outer_sector, pupil_sector)
    masked_image = cv2.bitwise_and(segmented, segmented, mask=mask)

    return masked_image, mask


def daugman_normalizaiton(image, height, width, r_in, r_out):
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height,width, 3), np.uint8)
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            print(i, j)
            color = image[int(Xc)][int(Yc)]  # color of the pixel

            flat[j][i] = color
    return flat  # liang


def crop_image(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]
