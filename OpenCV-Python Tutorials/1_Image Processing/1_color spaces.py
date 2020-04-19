import cv2
import numpy as np

# https://docs.opencv.org/3.1.0/df/d9d/tutorial_py_colorspaces.html
# http://colorizer.org/
img = cv2.imread("OpenCV-Python Tutorials/nemo0.jpg")

red = np.uint8([[[0, 0, 255]]])  # red in bgr
hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
print(hsv_red)

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsv, lower_red, upper_red)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("frame", img)
cv2.imshow("mask", mask)
cv2.imshow("res", res)


cv2.waitKey(0)
cv2.destroyAllWindows()
