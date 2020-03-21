import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from skimage.transform import resize
import random

im = data.page()
if len(im.shape) == 3:  # converting to gray scale if image is RGB
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # h 400 * w 600

fig = plt.figure(figsize=(30, 30))
rows = 2
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(im, cmap="gray")

out_thresholded = cv2.adaptiveThreshold(
    src=im,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=11,
    C=2,
)
fig.add_subplot(rows, columns, 2)
plt.imshow(out_thresholded, cmap="gray")
plt.show()
