# Your code will take an input tensor input with shape (n, iC, H, W) and a kernel kernel with shape (oC, iC, kH, kW). It needs then to apply a 2D convolution over input, using kernel as kernel tensor, using a stride of 1, no dilation, no grouping, and no padding, and store the result in out. Both input and kernel have dtype np.float32.

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


random.seed(1)
print("input")
n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
print("n  ", n, " iC  ", iC, " H  ", H, " W  ", W)
input = np.random.rand(n, iC, H, W)
print(input.shape)

print("\nkernel")
oC = random.randint(2, 6)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
print("oC  ", oC, " iC  ", iC, " kH  ", kH, " kW  ", kW)
kernel = np.random.rand(oC, iC, kH, kW)
print(kernel.shape)

pad = 0
stride = 1
dilation = 1
oH = ((H + (2 * pad) - (dilation * (kH - 1)) - 1) / stride) + 1
oW = ((W + (2 * pad) - (dilation * (kW - 1)) - 1) / stride) + 1
out = np.zeros(
    shape=(int(np.floor(n)), int(np.floor(oC)), int(np.floor(oH)), int(np.floor(oW))),
)
print("\nout")
print("n  ", n, " oC  ", oC, " oH  ", oH, " oW  ", oW)
print(out.shape)


def twoD_conv_single_step(input_slice, kernel_weights, bias=0):
    s = np.multiply(input_slice, kernel_weights)
    # Z = np.sum(s)
    Z = s.sum()
    # Z = Z + bias.astype(float)

    return Z


def twoD_conv_forward(input, kernel, output, pad, stride, dilation, bias=0):
    n, iC, H, W = input.shape
    oC, _, kH, kW = kernel.shape
    _, _, oH, oW = output.shape
    for N in range(n):
        input_sample = input[N, :, :, :]
        for OH in range(oH):
            count = 0
            for OW in range(oW):

                for OC in range(oC):

                    vertical_start = OH * stride
                    vertical_end = OH * stride + kH
                    horizontal_start = OW * stride
                    horizontal_end = OW * stride + kW

                    input_slice = input_sample[
                        :, vertical_start:vertical_end, horizontal_start:horizontal_end
                    ]
                    print(count, OC, OH, OW)
                    print(input_slice.shape, kernel[OC, :, :, :].shape)
                    output[N, OC, OH, OW] = twoD_conv_single_step(
                        input_slice, kernel[OC, :, :, :], bias=0
                    )
                    count = count + 1
    return output


out = twoD_conv_forward(input, kernel, out, pad, stride, dilation, bias=0)

# np.random.seed(1)
# a_slice_prev = np.random.randn(iC, H, W)
# W = np.random.randn(iC, kH, kW)
# b = np.random.randn(1, 1, 1)

# Z = twoD_conv_single_step(a_slice_prev, W, b)
# print("Z =", Z)

