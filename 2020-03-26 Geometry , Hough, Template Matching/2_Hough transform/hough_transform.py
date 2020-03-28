import numpy as np
import cv2
from skimage import data, feature
import matplotlib.pyplot as plt

# https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
# https://www.learnopencv.com/hough-transform-with-opencv-c-python/
# https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
# https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

fig = plt.figure(figsize=(30, 30))
rows = 2  # grid 2x2
columns = 2

im = data.coins()[160:230, 70:270]  # (70, 200)
if len(im.shape) == 3:  # converting to gray scale if image is RGB
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
h, w = im.shape
fig.add_subplot(rows, columns, 1)
plt.title("Original gray")
plt.imshow(im)  # cmap="gray"

edges = cv2.Canny(im, 230, 300)
fig.add_subplot(rows, columns, 2)
plt.title("Edges with Canny (cv2)")
plt.imshow(edges, cmap="gray")

edges_s = feature.canny(image=im, sigma=3, low_threshold=10, high_threshold=80)
fig.add_subplot(rows, columns, 4)
plt.title("Edges with Canny (scikit)")
plt.imshow(edges, cmap="gray")

# Apply hough transform on the image
circles = cv2.HoughCircles(
    image=edges,
    method=cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=15,
    param1=200,
    param2=10,
    minRadius=20,
    maxRadius=30,
)

# Draw detected circles
if circles is not None:
    circles = np.uint16(
        np.around(circles)
    )  ## convert the (x, y) coordinates and radius of the circles to integers
    for (x, y, r) in circles[0, :]:
        # Draw outer circle
        cv2.circle(img=im, center=(x, y), radius=r, color=(255, 255, 0), thickness=2)
        # Draw inner circle
        cv2.circle(img=im, center=(x, y), radius=1, color=(0, 0, 0), thickness=2)

fig.add_subplot(rows, columns, 3)
plt.title("With Hough")
plt.imshow(im)  # cmap="gray"

plt.show()
