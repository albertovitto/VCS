# Your code will take as input a grayscale image im (a np.ndarray with dtype np.uint8 and shape (H,W)). It needs then to:

#     Apply the horizontal and vertical Sobel masks (with a kernel size of 3) to obtain horizontal and vertical derivatives;
#     Compute the magnitude and direction of the gradient, and normalize them properly (see slides);
#     Diplay the gradient magnitude and derivative jointly in an HSV image, and then convert it in RGB format (see slides).

# The code is expected to show the final result using pyplot (e.g. calling the imshow function). When doing this, pay attention to the axis order.

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing

# from skimage image are RGB
im = data.coins()  # h 303 * w 384 (coins) or h 300 * w 451 * c 3 (chelsea)

fig = plt.figure(figsize=(20, 20))
rows = 2  # grid 2x3
columns = 3  # 3 images one next to the other
fig.add_subplot(rows, columns, 1)  # last num must be 1 <= num <= 6
plt.imshow(im)

if len(im.shape) == 3:  # converting to gray scale if image is RGB
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

im = np.swapaxes(im, 0, 1)  # h 384 * w 303

Gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)  # 3x3 sobel gradient over x 384*303
Gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)  # sobel gradient over y 384*303

M = np.sqrt(Gx ** 2 + Gy ** 2)  # V magnitude 384*303
max_M = np.sqrt((255 * 2 + 255 * 1) ** 2 + (255 * 1 + 255 * 2) ** 2)  # 1081.8
M = M / max_M  # 0-1
M = M * 255  # 0-255

theta = np.arctan2(Gy, Gx)  # H orientation/direction from -pi to +pi (384, 303)
theta = theta + np.pi  # 0 - 2pi
theta = ((theta * 180) / np.pi) / 2  # 0 - 180°

saturation = np.full_like(theta, 255)  # S (384, 303) value 255

HSV = np.array([theta, saturation, M])
# cv2.merge([h,s,v]) or np.stack([ , , ]) (3, 384, 303) dtype('float32')      values 0-255
HSV = HSV.astype(np.uint8)  # 32.24 -> 32
HSV = HSV.swapaxes(0, 2)  # (3, 384, 303) -> (303, 384, 3)

RGB = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)  # HSV -> RGB

fig.add_subplot(rows, columns, 2)  # pos 2
plt.imshow(np.swapaxes(im, 0, 1), cmap="gray")  # (384, 303) -> (303, 384)

fig.add_subplot(rows, columns, 3)
plt.imshow(np.swapaxes(M, 0, 1), cmap="gray")

fig.add_subplot(rows, columns, 4)  # pos 3
plt.imshow(RGB)  # (303, 384, 3)

fig.add_subplot(rows, columns, 5)
plt.imshow(np.swapaxes(Gx, 0, 1), cmap="gray")

fig.add_subplot(rows, columns, 6)
plt.imshow(np.swapaxes(Gy, 0, 1), cmap="gray")

plt.show()
