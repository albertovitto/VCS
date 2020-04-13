# Exercise
# : Given an input grayscale image im (a np.ndarray with shape (H, W) and dtype np.uint8), write a
# code which performs a binary thresholding of the image at cut value val , and stores the result in out
# ●
# out should be another image, with the same shape of im , and with all the pixels greater than the
# threshold set to 255, all the others set to 0
# ●
# Be careful not to modify the original tensor in place: the function should perform the thresholding on
# a copy of the image

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from skimage.transform import resize
import random

# random.seed(0)

im = data.coffee()  # h 400 * w 600 * c 3
if len(im.shape) == 3:  # converting to gray scale if image is RGB
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # h 400 * w 600

fig = plt.figure(figsize=(30, 30))
rows = 2  # grid 2x3
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

val = random.randint(0, 255)
print(val)

out = np.copy(im)
out = (out > val).astype(np.uint8) * 255
# (out > val) -> matrix of true and false with size of out
# .astype(np.uint8) convert true to 1, false to 0
# * 255 convert 1 to 255, 0 to 0 so it's drawable
out = np.swapaxes(out, 0, 1)
fig.add_subplot(rows, columns, 4)
plt.imshow(out, cmap="gray")

out2 = np.copy(im)
out2[out2 > val] = 255  # where out2 is over val set 255
out2[out2 != 255] = 0  # all others cases 0
out2 = np.swapaxes(out2, 0, 1)
fig.add_subplot(rows, columns, 5)
plt.imshow(out2, cmap="gray")

plt.show()
