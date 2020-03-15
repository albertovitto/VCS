# Your code will take an input tensor input with shape (n, iC, H, W) and a kernel kernel with shape (oC, iC, kH, kW). It needs then to apply a 2D convolution over input, using kernel as kernel tensor, using a stride of 1, no dilation, no grouping, and no padding, and store the result in out. Both input and kernel have dtype np.float32.

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing


n = input.shape[0]
iC = input.shape[1]
H = input.shape[2]
W = input.shape[3]
print(n, iC, H, W)
print(input.shape)

oC = kernel.shape[0]
iC = kernel.shape[1]
kH = kernel.shape[2]
kW = kernel.shape[3]
print(oC, iC, kH, kW)
print(kernel.shape)

pad = 0
stride = 1
dilation = 1
oH = ((H + (2 * pad) - (dilation * (kH - 1)) - 1) / stride) + 1
oW = ((W + (2 * pad) - (dilation * (kW - 1)) - 1) / stride) + 1
out = np.zeros(
    shape=(int(np.floor(n)), int(np.floor(oC)), int(np.floor(oH)), int(np.floor(oW))),
)
print(out.shape)


def twoD_conv_single_step(input_slice, kernel_weights, bias=0):
    s = np.multiply(input_slice, kernel_weights)
    # Z = np.sum(s)
    Z = s.sum()
    Z = Z + bias.astype(float)

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
