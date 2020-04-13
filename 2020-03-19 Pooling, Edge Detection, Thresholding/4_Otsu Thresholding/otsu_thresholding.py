# Exercise
# : Given an input grayscale image im (a np.ndarray with shape (H, W) and dtype np.uint8), write a
# code which computes the Otsu threshold for im and stores the result in out
# ●
# You are not allowed to use any Python package except from Numpy
# ●
# Notice: beware of how the threshold is defined in the formulas of the previous slides. Your output
# should be compliant with our definition of threshold (first slide).

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#otsus-binarization

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skimage.transform import resize
import random

# try page()
im = data.grass()  # h 400 * w 600 * c 3
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

thr_value, thr_out = cv2.threshold(im, 127, 255, cv2.THRESH_OTSU)
print("Real thr: ", thr_value)
thr_out = np.swapaxes(thr_out, 0, 1)
fig.add_subplot(rows, columns, 5)
plt.imshow(thr_out, cmap="gray")

# histogram_array = cv2.calcHist(
#     images=im, channels=[0], mask=None, histSize=[256], ranges=[0, 256]
# )
N = 256
histogram, bin_edges = np.histogram(
    a=im.ravel(), bins=N, range=[0, N]
)  # histogram from 0 to 255 : [0, 0, 1, 2, 35, 60, ...
fig.add_subplot(rows, columns, 6)
plt.hist(x=im.ravel(), bins=N, range=[0, N], density=True, stacked=True)

bins = np.arange(N)  # [0, 1, ...., 255]
best_threshold = -np.inf  # can go 0 - 255
max_intra_class_variance = -np.inf
for t in range(N):  # 0 to 255
    w1 = np.sum(histogram[: (t + 1)])  # from 0 included to t+1 excluded
    w2 = np.sum(histogram[(t + 1) : N])
    if w1 != 0 and w2 != 0:
        u1 = np.sum(histogram[: (t + 1)] * bins[: (t + 1)]) / w1  # weighted sum
        u2 = np.sum(histogram[(t + 1) : N] * bins[(t + 1) : N]) / w2
        intra_class_variance = w1 * w2 * (u1 - u2) ** 2
        if intra_class_variance > max_intra_class_variance:
            max_intra_class_variance = intra_class_variance
            best_threshold = t
print("Computed thr: ", best_threshold)

out = np.copy(im)
out = (out > best_threshold).astype(np.uint8) * 255
out = np.swapaxes(out, 0, 1)
fig.add_subplot(rows, columns, 8)
plt.imshow(out, cmap="gray")

plt.show()
