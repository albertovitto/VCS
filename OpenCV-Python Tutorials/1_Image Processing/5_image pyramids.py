import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2
import numpy as np, sys

A = cv2.imread("OpenCV-Python Tutorials/apple.jpg")
B = cv2.imread("OpenCV-Python Tutorials/orange.jpg")

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i - 1], GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)


def floor_and_int(n):
    return int(np.floor(n))


# Now add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack(
        (la[:, 0 : floor_and_int(cols / 2)], lb[:, floor_and_int(cols / 2) :])
    )
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:, : floor_and_int(cols / 2)], B[:, floor_and_int(cols / 2) :]))

cv2.imshow("Pyramid_blending2.jpg", ls_)

cv2.imshow("Direct_blending.jpg", real)

cv2.waitKey(0)
cv2.destroyAllWindows()
