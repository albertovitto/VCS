import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread("OpenCV-Python Tutorials/ml.png")
img2 = cv2.imread("OpenCV-Python Tutorials/opencv-logo.png")

height = 380
width = 308
dim = (width, height)

# resize image
resized = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
print(img1.shape, resized.shape)
dst = cv2.addWeighted(img1, 0.7, resized, 0.3, 0)
dst1 = cv2.add(img1 * 0.7, resized * 0.3 + 0)  # saturate in 0-255
dst1 = dst1.astype(np.uint8)

cv2.imshow("dst", dst)
cv2.imshow("dst1", dst1)
cv2.waitKey(0)
cv2.destroyAllWindows()
