import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("OpenCV-Python Tutorials/j.png", 0)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
erosion_dilation = cv2.dilate(erosion, kernel, iterations=1)
dilation_erosion = cv2.erode(dilation, kernel, iterations=1)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

titles = [
    "J",
    "J ERODED",
    "J DILATED",
    "J ERODED AND DILATED",
    "J DILATED AND ERODED",
    "DIFFERENCE DILATION-EROSION",
]
images = [img, erosion, dilation, erosion_dilation, dilation_erosion, gradient]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
