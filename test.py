import cv2
import numpy as np


img = cv2.imread('./imagesss/iris5.jpg', 0)
h, w = img.shape[:2]
final = np.zeros((h, w, 3), np.uint8)





_, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold', thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [cv2.approxPolyDP(cnt, 1, True) for cnt in contours]

'''
(x,y),radius = cv2.minEnclosingCircle(contours[0])
center = (int(x),int(y))
radius = int(radius)
cv2.circle(img, center, radius, (0, 255, 0), 2)
cv2.imshow('sdadsae', img)
'''

levels = 0
cv2.drawContours(final, contours, -1, (128,255,255), 3, cv2.LINE_AA)

final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
cv2.imshow('contours', final)


circles = cv2.HoughCircles(
    final, cv2.HOUGH_GRADIENT, 1.8, img.shape[0], param1=50, param2=20, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (255, 255, 0), 3)


cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()