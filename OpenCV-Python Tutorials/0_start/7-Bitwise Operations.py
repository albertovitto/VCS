import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load two images
img1 = cv2.imread("OpenCV-Python Tutorials/messi5.jpg")  # (342, 548, 3)
cv2.imshow("img1", img1)
img2 = cv2.imread("OpenCV-Python Tutorials/opencv-logo-white.png")  # (222, 180, 3)
cv2.imshow("img2", img2)

# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]  # (222, 180, 3)
cv2.imshow("roi", roi)

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # convert to gray logo
ret, mask = cv2.threshold(
    img2gray, 0, 255, cv2.THRESH_BINARY
)  # where there is pixel > 10 -> 255 w, if < 10 -> 0 b
cv2.imshow("mask", mask)
mask_inv = cv2.bitwise_not(mask)  # 0->255 255->0   (222, 180)
cv2.imshow("mask_inv", mask_inv)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(
    roi, roi, mask=mask_inv
)  # create black background over roi in order to easily put final logo on it (23 and 0) = 0, (23 and 255) = 23
cv2.imshow("img1_bg", img1_bg)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)  # (23 and 0) = 0, (23 and 255) = 23
cv2.imshow("img2_fg", img2_fg)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
cv2.imshow("dst", dst)

img1[0:rows, 0:cols] = dst
cv2.imshow("res", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
