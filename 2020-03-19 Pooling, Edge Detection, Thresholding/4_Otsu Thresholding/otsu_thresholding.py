# Exercise
# : Given an input grayscale image im (a np.ndarray with shape (H, W) and dtype np.uint8), write a
# code which computes the Otsu threshold for im and stores the result in out
# ●
# You are not allowed to use any Python package except from Numpy
# ●
# Notice: beware of how the threshold is defined in the formulas of the previous slides. Your output
# should be compliant with our definition of threshold (first slide).

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from skimage.transform import resize
import random


im = data.coffee()  # h 400 * w 600 * c 3
if len(im.shape) == 3:  # converting to gray scale if image is RGB
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # h 400 * w 600

fig = plt.figure(figsize=(30, 30))
rows = 3  # grid 3x3
columns = 3
fig.add_subplot(rows, columns, 1)
plt.imshow(im, cmap="gray")


im = resize(
    im,
    (im.shape[0] // 2, im.shape[1] // 2),
    mode="reflect",
    preserve_range=True,
    anti_aliasing=True,
).astype(
    np.uint8
)  # h 200 * w 300
fig.add_subplot(rows, columns, 2)
plt.imshow(im, cmap="gray")

im = np.swapaxes(im, 0, 1)  # h 300 * w 200
fig.add_subplot(rows, columns, 3)
plt.imshow(im, cmap="gray")


out = np.copy(im)
out = np.swapaxes(out, 0, 1)
fig.add_subplot(rows, columns, 4)
plt.imshow(out, cmap="gray")

thr_value, thr_out = cv2.threshold(im, 127, 255, cv2.THRESH_OTSU)
print(thr_value)
thr_out = np.swapaxes(thr_out, 0, 1)
fig.add_subplot(rows, columns, 5)
plt.imshow(thr_out, cmap="gray")

# histogram_array = cv2.calcHist(
#     images=im, channels=[0], mask=None, histSize=[256], ranges=[0, 256]
# )
histogram, bin_edges = np.histogram(a=im.ravel(), bins=256, range=[0, 256],)
fig.add_subplot(rows, columns, 6)
plt.hist(x=im.ravel(), bins=256, range=[0, 256], density=True, stacked=True)
b = 0


plt.show()
