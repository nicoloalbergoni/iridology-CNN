import cv2
import numpy as np

image = cv2.imread('iris1.jpg')
# res = cv2.resize(image, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

circles = cv2.HoughCircles(
    thresh, cv2.HOUGH_GRADIENT, 1, image.shape[0], param1=30, param2=20, minRadius=0, maxRadius=0)


circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Threshold', thresh)
cv2.imshow('Detected circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
