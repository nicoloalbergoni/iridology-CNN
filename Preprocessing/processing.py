import cv2
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from Preprocessing.display import draw_ellipse
from Preprocessing.filtering import filtering, threshold


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

    if circles is None:
        # TODO: Gestire il caso in cui non trova cerchi
        pass
    elif circles.shape[0:2] == (1, 1):
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

    if circles is None:
        # TODO: Gestire il caso in cui non trova cerchi
        pass
    elif circles.shape[0:2] == (1, 1):
        return circles[0, 0]
    else:
        # TODO: Gestire il caso in cui trovo più cerchi
        pass


def segmentation(image, iris_circle, pupil_circle, startangle, endangle):
    segmented = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = segmented.shape
    outer_sector = np.zeros((height, width), np.uint8)
    pupil_sector = np.zeros((height, width), np.uint8)
    draw_ellipse(outer_sector, (iris_circle[0], iris_circle[1]), (
        iris_circle[2], iris_circle[2]), 0, -startangle, -endangle, 255, thickness=-1)
    cv2.circle(pupil_sector, (pupil_circle[0], pupil_circle[1]), int(
        pupil_circle[2]), 255, thickness=-1)
    mask = cv2.subtract(outer_sector, pupil_sector)
    masked_image = cv2.bitwise_and(segmented, segmented, mask=mask)

    return masked_image, mask


def crop_image(masked_image, offset=30, tollerance=80):
    mask = masked_image > tollerance

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = masked_image[x0 - offset: x1 + offset, y0 - offset: y1 + offset]
    #print('Cropped Image Shape', cropped.shape)

    return cropped


def daugman_normalizaiton(original_eye, circle, pupil_radius=0, startangle=0, endangle=45):
    
    
    start_angle = (360 - endangle) * np.pi / 180
    end_angle = (360 - startangle) * np.pi / 180
    
    iris_coordinates = (circle[0], circle[1])
    
    
    x = int(iris_coordinates[0])
    y = int(iris_coordinates[1])

    w = int(round(circle[2]) + 0)
    h = int(round(circle[2]) + 0)

    #cv2.circle(original_eye, iris_coordinates, int(circle[2]), (255,0,0), thickness=2)
    iris_image = original_eye[y-h:y+h,x-w:x+w]
    
    
    iris_image_to_show = cv2.resize(iris_image, (iris_image.shape[1]*2, iris_image.shape[0]*2))

    q = np.arange(start_angle, end_angle, 0.01) #theta
    inn = np.arange(int(pupil_radius), int(iris_image_to_show.shape[0]/2), 1) #radius

    cartisian_image = np.empty(shape = [inn.size, int(iris_image_to_show.shape[1])])
    m = interp1d([np.pi*2, 0],[pupil_radius,iris_image_to_show.shape[1]])

    for r in tqdm(inn):
        for t in tqdm(q):
            polarX = int((r * np.cos(t)) + iris_image_to_show.shape[1]/2)
            polarY = int((r * np.sin(t)) + iris_image_to_show.shape[0]/2)
            cartisian_image[r][int(m(t) - 1)] = iris_image_to_show[polarY][polarX]
    
    cartisian_image = cartisian_image.astype('uint8')     
    return cartisian_image