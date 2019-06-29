import cv2
import keyboard
import numpy as np

closeWindows = False


def show_images(image):
    """
    Activate image display

    :param image: image with the circles found
    :type image: numpy.ndarray
    :rtype: None
    """
    global closeWindows
    if closeWindows is True:
        return
    cv2.imshow('Circled', image)
    cv2.waitKey(0)
    if keyboard.is_pressed('esc'):
        closeWindows = True
        cv2.destroyAllWindows()
        return
    cv2.destroyAllWindows()


def draw_circles(real_image, pupil_circles, iris_circles):
    """
    Draws iris and pupil circles onto the original image

    :param real_image: Original image
    :type real_image: numpy.ndarray
    :param pupil_circles: array indicating center and radius of the pupil circle
    :type pupil_circles: numpy.ndarray
    :param iris_circles: array indicating center and radius of the iris circle
    :type iris_circles: numpy.ndarray
    :return: None
    """
    if pupil_circles is not None:
        pupil_circles = np.uint16(np.around(pupil_circles))
        cv2.circle(
            real_image, (pupil_circles[0], pupil_circles[1]), pupil_circles[2], (255, 0, 0), 2)
        cv2.circle(
            real_image, (pupil_circles[0], pupil_circles[1]), 2, (255, 255, 0), 3)

    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        cv2.circle(
            real_image, (iris_circles[0], iris_circles[1]), iris_circles[2], (0, 255, 0), 2)
        cv2.circle(
            real_image, (iris_circles[0], iris_circles[1]), 2, (0, 0, 255), 3)


def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=1, lineType=cv2.LINE_AA, shift=10):
    """
    Draws a section of ellipse with a given center, axes, and angles

    :param img: image on which to draw
    :type img: numpy.ndarray
    :param center: Tuple that contains the coordinates of the center of the ellipse
    :type center: Tuple[numpy.float32, numpy.float32]
    :param axes: Tuple that contains half the size of the two axes of the ellipse
    :type axes: Tuple[numpy.float64, numpy.float64]
    :param angle: Ellipse rotation angle in degrees
    :type angle: int
    :param startAngle: Starting angle of the elliptic arc in degrees
    :type startAngle: int
    :param endAngle: Ending angle of the elliptic arc in degrees
    :type endAngle: int
    :param color: Color of the line to be drawn
    :type color: int
    :param thickness: Thickness of the line to be drawn
    :type thickness: int
    :param lineType:  Type of the ellipse boundary
    :type lineType: int
    :param shift: Number of fractional bits in the coordinates of the center and values of axes
    :type shift: int
    """
    center = (
        int(round(center[0] * 2 ** shift)),
        int(round(center[1] * 2 ** shift))
    )
    axes = (
        int(round(axes[0] * 2 ** shift)),
        int(round(axes[1] * 2 ** shift))
    )
    return cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)
