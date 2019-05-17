import cv2
import numpy as np
from display import show_images


def filtering(img, invgray=False, sharpen=False, grayscale=True):
    frame = img

    if grayscale is True:
        cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        cimg = frame

    if invgray is True:
        cimg = cv2.bitwise_not(cimg)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blackhat = cv2.morphologyEx(cimg, cv2.MORPH_BLACKHAT, kernel)
    bottom_hat_filtered = cv2.add(blackhat, cimg)
    # final_img = cv2.medianBlur(bottom_hat_filtered, 17)
    # final_img = cv2.blur(bottom_hat_filtered, (3, 3))

    final_img = cv2.GaussianBlur(bottom_hat_filtered, (5, 5), 0)

    #final_img = cv2.bilateralFilter(cimg, 9, 75, 75)

    if sharpen is True:
        # ---Sharpening filter----
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        final_img = cv2.filter2D(final_img, -1, kernel)

        kernel = np.ones((5, 5), np.float32)/25
        final_img = cv2.filter2D(final_img, -1, kernel)
    return final_img


def adjust_gamma(image, gamma=1.0):
    """
    Building a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values. Increasing contrast
    :param image: image
    :param gamma: adjusting coefficient
    :return: adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_v = v.mean()

    print(mean_v)

    if mean_v > 165:
        gamma = 0.7
    elif mean_v < 155:
        gamma = 1.3

    hsv = cv2.merge((h, s, v))

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def threshold(img, tValue=100, adaptive=False, binaryInv=False, otsu=False, dilate=False):
    if adaptive is False and otsu is False:
        _, thresh = cv2.threshold(img, tValue,
                                  255, cv2.THRESH_BINARY_INV if binaryInv else cv2.THRESH_BINARY)
    elif adaptive is True and otsu is False:
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1)
    elif adaptive is False and otsu is True:
        ret, thresh = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        thresh = img

    if dilate is True:
        thresh = dilate_thresh(thresh)

    return thresh


def dilate_thresh(thresh):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh_img = cv2.dilate(opening, kernel, iterations=3)

    return thresh_img


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_v = v.mean()

    print(mean_v)

    if mean_v < 140:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        hsv = cv2.merge((h, s, v))

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img
