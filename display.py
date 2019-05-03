import cv2
import numpy as np


def show_images(image):
    cv2.imshow('Circled', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_circles(real_image, pupil_circles, iris_circles):
    if pupil_circles is not None:
        pupil_circles = np.uint16(np.around(pupil_circles))
        for i in pupil_circles[0, :]:
            cv2.circle(real_image, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.circle(real_image, (i[0], i[1]), 2, (255, 255, 0), 3)

    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        for i in iris_circles[0, :]:
            cv2.circle(real_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(real_image, (i[0], i[1]), 2, (0, 0, 255), 3)
