import cv2 as cv
import numpy as np

img = cv.imread("2020-03-05/img.png")
width, height, channels = img.shape
new = cv.convertScaleAbs(img, alpha=-11, beta=0)
for x in range(0, width):
    for y in range(0, height):
        # img[x, y] = img[x, y] + img[x, y] * 0.1
        # img[x, y] = np.clip(img[x, y] + -1.1 * img[x, y] + 0, 0, 255)
        a = 4

cv.imshow("new", img)
cv.waitKey()
cv.imshow("new", new)
cv.waitKey()
