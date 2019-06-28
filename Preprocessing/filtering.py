import cv2
import numpy as np


def filtering(img, invgray=False, sharpen=False, grayscale=True):
    """
    Apply several filters to an image, such as: grayscale, Bottom Hat Filter and Gaussian Blur.
    It also provides the ability to enable two more filters: Inverted grayscale and Sharpen.

    :param img: Image to be filtered
    :type img: numpy.ndarray
    :param invgray: if true enables the Inverted Grayscale filter
    :type invgray: bool
    :param sharpen: if true enables the Sharpen filter
    :type sharpen: bool
    :param grayscale: if true converts the image in grayscale
    :type grayscale: bool
    :return: The image with the selected filters applied
    :rtype: numpy.ndarray
    """
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
    final_img = cv2.GaussianBlur(bottom_hat_filtered, (5, 5), 0)

    if sharpen is True:
        # ---Sharpening filter----
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        final_img = cv2.filter2D(final_img, -1, kernel)

        kernel = np.ones((5, 5), np.float32)/25
        final_img = cv2.filter2D(final_img, -1, kernel)
    return final_img


def adjust_gamma(image, gamma=1.0):
    """
    Increase contrast by building a lookup table mapping to map the pixel values [0, 255] to
    their adjusted gamma values.

    :param image: image
    :type image: numpy.ndarray
    :param gamma: adjusting coefficient
    :type gamma: float
    :return: adjusted image
    :rtype: numpy.ndarray
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_v = v.mean()

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


def threshold(img, tValue, adaptive=False, binaryInv=False, otsu=False, dilate=False):
    """
    Apply one threshold method to the image based on the chosen parameter.
    By default the function apply a binary threshold to the image.

    :param img: image
    :type img: numpy.ndarray
    :param tValue: threshold value
    :type tValue: int
    :param adaptive: if true apply an adaptive threshold method
    :type adaptive: bool
    :param binaryInv: if true apply an inverted binary threshold
    :type binaryInv: bool
    :param otsu: if true apply OTSU's algorithm to find the optimal threshold value then perform an inverted
        binary threshold with the calculated threshold value
    :type otsu: bool
    :param dilate: if true apply a dilatation transformation to the image after performing one of the
        previous threshold methods
    :type dilate: bool
    :return: the binarized image
    :rtype: numpy.ndarray
    """
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
    """
    Apply a morphological dilatation to the image

    :param thresh: resulted image of a threshold method
    :type thresh: numpy.ndarray
    :return: transformed image
    :rtype: numpy.ndarray
    """
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh_img = cv2.dilate(opening, kernel, iterations=3)

    return thresh_img


def increase_brightness(img, value=30):
    """
    Increase the brightness of an image

    :param img: image
    :type img: numpy.ndarray
    :param value: brightness factor
    :type value: int
    :return: brighten image
    :rtype: numpy.ndarray
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_v = v.mean()

    if mean_v < 140:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        hsv = cv2.merge((h, s, v))

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img
