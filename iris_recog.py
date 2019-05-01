import cv2
import numpy as np


def load_image(path, grayscale=False, blur=False):
    global image
    global real_image
    image = cv2.imread(path)
    real_image = image

    if grayscale is True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur is True:
        image = cv2.blur(image, (5, 5))
        #image = cv2.medianBlur(image, ksize=5)
        cv2.imshow('Blurred image', image)


def draw_circles(circles, pupil=False):
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:

        if pupil is False:
            # draw the outer circle
            cv2.circle(real_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(real_image, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            cv2.circle(real_image, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.circle(real_image, (i[0], i[1]), 2, (255, 255, 0), 3)


def display_image():
    cv2.imshow('Detected iris + pupil', real_image)


def iris_recognition():
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 1, image.shape[0], param1=30, param2=20, minRadius=0, maxRadius=0)
    draw_circles(circles)


def pupil_recognition():
    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 1, image.shape[0], param1=30, param2=20, minRadius=0, maxRadius=0)
    draw_circles(circles, pupil=True)
    #cv2.imshow('Threshold pupil', thresh)


load_image('./images/iris3.jpg', grayscale=True, blur=True)
iris_recognition()
pupil_recognition()

display_image()
cv2.waitKey(0)
cv2.destroyAllWindows()
