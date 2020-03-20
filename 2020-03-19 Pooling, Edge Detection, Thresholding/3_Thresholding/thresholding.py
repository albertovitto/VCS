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

im = data.camera()
im = resize(
    im,
    (im.shape[0] // 2, im.shape[1] // 2),
    mode="reflect",
    preserve_range=True,
    anti_aliasing=True,
).astype(np.uint8)
im = np.swapaxes(im, 0, 1)
val = random.randint(0, 255)
