# Your code will take an input tensor input with shape (n, iC, H, W) and a kernel kernel with shape (oC, iC, kH, kW). It needs then to apply a 2D convolution over input, using kernel as kernel tensor, using a stride of 1, no dilation, no grouping, and no padding, and store the result in out. Both input and kernel have dtype np.float32.

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import random


n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
print(n, iC, H, W)

oC = random.randint(2, 6)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
print(oC, iC, kH, kW)

input = np.random.rand(n, iC, H, W)
print(input.shape)

kernel = np.random.rand(oC, iC, kH, kW)
print(kernel.shape)

