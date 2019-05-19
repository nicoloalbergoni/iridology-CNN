import cv2
import numpy as np


def show_images(image):
    cv2.imshow('Circled', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_circles(real_image, pupil_circles, iris_circles):
    if pupil_circles is not None:
        pupil_circles = np.uint16(np.around(pupil_circles))
        cv2.circle(real_image, (pupil_circles[0], pupil_circles[1]), pupil_circles[2], (255, 0, 0), 2)
        cv2.circle(real_image, (pupil_circles[0], pupil_circles[1]), 2, (255, 255, 0), 3)

    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        cv2.circle(real_image, (iris_circles[0], iris_circles[1]), iris_circles[2], (0, 255, 0), 2)
        cv2.circle(real_image, (iris_circles[0], iris_circles[1]), 2, (0, 0, 255), 3)

def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=1, lineType=cv2.LINE_AA, shift=10):
    # uses the shift to accurately get sub-pixel resolution for arc
    # taken from https://stackoverflow.com/a/44892317/5087436
    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    return cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)

