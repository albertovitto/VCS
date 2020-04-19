import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("OpenCV-Python Tutorials/sudoku.png", 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian_abs = np.absolute(laplacian)
laplacian_uin8 = np.uint8(laplacian_abs)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobelx_abs = np.absolute(sobelx)
sobelx_uin8 = np.uint8(sobelx_abs)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobely_abs = np.absolute(sobely)
sobely_uin8 = np.uint8(sobely_abs)

rows, cols = 3, 4
plt.subplot(rows, cols, 1), plt.imshow(img, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 2), plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian"), plt.xticks([]), plt.yticks([])
plt.subplot(rows, cols, 3), plt.imshow(laplacian_abs, cmap="gray")
plt.title("Laplacian abs"), plt.xticks([]), plt.yticks([])
plt.subplot(rows, cols, 4), plt.imshow(laplacian_uin8, cmap="gray")
plt.title("Laplacian uin8"), plt.xticks([]), plt.yticks([])


plt.subplot(rows, cols, 6), plt.imshow(sobelx, cmap="gray")
plt.title("Sobel X"), plt.xticks([]), plt.yticks([])
plt.subplot(rows, cols, 7), plt.imshow(sobelx_abs, cmap="gray")
plt.title("Sobel X abs"), plt.xticks([]), plt.yticks([])
plt.subplot(rows, cols, 8), plt.imshow(sobelx_uin8, cmap="gray")
plt.title("Sobel X ui8"), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 10), plt.imshow(sobely, cmap="gray")
plt.title("Sobel Y"), plt.xticks([]), plt.yticks([])

plt.show()
