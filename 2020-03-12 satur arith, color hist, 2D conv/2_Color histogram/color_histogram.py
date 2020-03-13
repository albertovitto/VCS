import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing

# Your code will take as input a color image im (a np.ndarray with dtype np.uint8 and shape (3, H, W)) and an integer nbin. It should compute a normalized color histogram of the image, quantized with nbin bins on each color plane.

# The output should be a np.ndarray with shape (3*nbin, ), containing the concatenation of the histograms computed on each color plane (in the same order of the input tensor).

# The output should be L1-normalized (i.e. all bins of the final histogram should sum up to 1).

# Quantization strategy: a pixel should go in the bin with index b iif: pixel*n_bin // 256 == b


im = cv2.imread(
    "2020-03-12 satur arith, color hist, 2D conv/2_Color histogram/ast.png",
    cv2.IMREAD_COLOR,
)  # BGR H W 3 (400, 600, 3)
# print(im.shape)
# cv2.imshow("img", im)
# cv2.waitKey()
im = im.swapaxes(0, 2)  # BGR 3 W H (3, 600, 400)
# print(im.shape)

nbin = 35
color_histogram = []

for c in range(im.shape[0]):
    histogram = np.zeros(shape=(nbin,))
    for w in range(im.shape[1]):
        for h in range(im.shape[2]):
            pixel = im[c, w, h]  # scalar
            bin = pixel * nbin // 256  # 123*42 // 256 = 20
            histogram[bin] += 1
    color_histogram = np.concatenate((color_histogram, histogram), axis=None)
    # a = c * nbin
    # b = (c + 1) * nbin
    # color_histogram[a:b] = histogram
    # print(np.sum(color_histogram))

out = color_histogram / np.sum(color_histogram)
# out = color_histogram / (im.shape[0] * im.shape[1] * im.shape[2]) equivalent
print((im.shape[0] * im.shape[1] * im.shape[2]), np.sum(color_histogram))
print(np.sum(out))
