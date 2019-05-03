import cv2
import numpy as np
from display import show_images


def filtering(img, invgray=False):
    frame = img

    cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if invgray is True:
        cimg = cv2.bitwise_not(cimg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blackhat = cv2.morphologyEx(cimg, cv2.MORPH_BLACKHAT, kernel)
    bottom_hat_filtered = cv2.add(blackhat, cimg)
    #final_img = cv2.medianBlur(bottom_hat_filtered, 17)
    final_img = cv2.blur(bottom_hat_filtered, (3, 3))
    return final_img


def adjust_gamma(image, gamma=1.0):
    """
    Building a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values. Increasing contrast
    :param image: image
    :param gamma: adjusting coefficient
    :return: adjusted image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
