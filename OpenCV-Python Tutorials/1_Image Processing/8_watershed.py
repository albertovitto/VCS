import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("OpenCV-Python Tutorials/coins.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

rows, cols = 3, 3
plt.subplot(rows, cols, 1), plt.imshow(gray, cmap="gray")
plt.title("Input gray Image"), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 2), plt.imshow(thresh, cmap="gray")
plt.title("thresh"), plt.xticks([]), plt.yticks([])

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

plt.subplot(rows, cols, 3), plt.imshow(sure_bg, cmap="gray")
plt.title("sure_bg"), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 4), plt.imshow(sure_fg, cmap="gray")
plt.title("sure_fg"), plt.xticks([]), plt.yticks([])

plt.subplot(rows, cols, 5), plt.imshow(dist_transform, cmap="gray")
plt.title("dist_transform"), plt.xticks([]), plt.yticks([])

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = markers.astype(np.uint8)
# jet = cv2.applyColorMap(markers, colormap=cv2.COLORMAP_JET)
# plt.subplot(rows, cols, 6), plt.imshow(jet)
# plt.title("jet"), plt.xticks([]), plt.yticks([])

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

plt.show()
