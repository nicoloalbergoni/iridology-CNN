import cv2
import numpy as np
import os


def load_image(path):
    global images
    global real_images
    global images_names
    images = []
    images_names = []
    for file in os.listdir(path):
        title = file.title().lower()
        if title.split('.')[-1] == 'jpg':
            images_names.append(title)
            images.append(cv2.imread(os.path.join(path, title)))
    real_images = images


def filtering():
    global filtered
    filtered = []
    for img in images:
        cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blackhat = cv2.morphologyEx(cimg, cv2.MORPH_BLACKHAT, kernel)
        bottom_hat_filtered = cv2.add(blackhat, cimg)
        final_img = cv2.medianBlur(bottom_hat_filtered, 17)
        #final_img = cv2.blur(cimg,(3,3))
        filtered.append(final_img)
    show_images_array(filtered)
    return filtered


def show_images_array(my_images):
    for img in my_images:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def iris_recognition():
    for img in filtered:
        _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            thresh, cv2.HOUGH_GRADIENT, 1, img.shape[0], param1=30, param2=20, minRadius=0, maxRadius=0)
        draw_circles(circles)


def draw_circles(circles, pupil=False):
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        if pupil is False:
            # draw the outer circle
            cv2.circle(real_images, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(real_images, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            cv2.circle(real_images, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.circle(real_images, (i[0], i[1]), 2, (255, 255, 0), 3)


def pupil_recognition():
    for img in filtered:
        _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            thresh, cv2.HOUGH_GRADIENT, 1, img.shape[0], param1=30, param2=20, minRadius=0, maxRadius=0)
        draw_circles(circles, pupil=True)
    #cv2.imshow('Threshold pupil', thresh)


def display_image():
    cv2.imshow('Detected iris + pupil', real_images)


load_image('C:\\Users\\Albe\\Desktop\\tesi-triennale\\images')
filtering()
iris_recognition()
pupil_recognition()

display_image()
cv2.waitKey(0)
cv2.destroyAllWindows()
