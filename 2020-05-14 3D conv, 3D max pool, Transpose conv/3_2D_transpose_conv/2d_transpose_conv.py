# Your code will take an input tensor input with shape (n, iC, H, W), a kernel kernel with shape (iC, oC, kH, kW) and a stride s.

# It needs then to apply a 2D Transpose convolution over input, using kernel as kernel tensor, using a stride of s on both axes, no dilation, no grouping, and no padding, and store the result in out.

# input and kernel are torch.Tensor with dtype torch.float32. s is an integer.
# http://d2l.ai/chapter_computer-vision/transposed-conv.html#fig-trans-conv
# https://distill.pub/2016/deconv-checkerboard/

import numpy as np
import random
import torch
import time


np.random.seed(0)
random.seed(1)
torch.manual_seed(0)


n = 2  # random.randint(2, 6)
iC = 3  # random.randint(2, 6)
oC = 6  # random.randint(2, 6)
H = 13  # random.randint(10, 20)
W = 20  # random.randint(10, 20)
kH = 3  # random.randint(2, 6)
kW = 3  # random.randint(2, 6)

input = torch.rand(n, iC, H, W)
kernel = torch.rand(iC, oC, kH, kW)
print("in   n:{}   iC:{}   H:{}    W:{}".format(n, iC, H, W))
print("ke   iC:{}  oC:{}  kH:{}   kW:{}".format(iC, oC, kH, kW))


pad = 0
stride = 5  # random.randint(2, 6)
dilation = 1
print("st     :{}".format(stride))

oH = np.int((H - 1) * stride - 2 * pad + dilation * (kH - 1) + 1)
oW = np.int((W - 1) * stride - 2 * pad + dilation * (kW - 1) + 1)

out = torch.zeros(n, oC, oH, oW)
print("ou   n:{}  oC:{}   oH:{}   oW:{}".format(n, oC, oH, oW))
out = out.to(torch.float32)


start = time.time()
# 4 for loop
for n_ in range(n):  # for each input sample
    for h in range(H):
        for w in range(W):  # for each pixel identified by h,w
            for oc in range(oC):  # for each output channel = # kernel I have
                input_slice = input[n_, :, h, w]
                # ([2, 3, 13, 20]) -> ([3])
                # from the input I take a sample, a pixel (h,w) and all his channels (:)
                input_slice.unsqueeze_(dim=1).unsqueeze_(dim=2)
                # ([3]) -> ([3, 1]) -> ([3, 1, 1])
                # then I unsqueeze it to make it fit with kernel
                kernel_ = kernel[:, oc, :, :]
                # ([3, 6, 3, 3]) -> ([3, 3, 3])
                # I take all the channels and dimensions of one kernel
                product = torch.sum(input=(input_slice * kernel_), dim=(0))
                # ([3, 1, 1]) *
                # ([3, 3, 3]) ->
                # ([3, 3, 3]) ->
                # ([   3, 3])
                # then I multiply a matrix 3x3 with a cube 3x3x3 (each level of the cube is element-wise multiplied), then I compress the first axis, the one of the channels, by sum over it
                out[
                    n_, oc, h * stride : h * stride + kH, w * stride : w * stride + kW
                ] += product
                # out[...] is ([3, 3]) portion, insert that piece in final output with right stride, if cell was already I add the new value and sum all together
end = time.time()
print("Looping over n, H, W, oC ", end - start)


start = time.time()
# try to make a faster method through broadcasting, compress N sample
for h in range(H):
    for w in range(W):  # for each pixel identified by h,w
        for oc in range(oC):  # for each output channel = kernel I have
            input_slice = input[:, :, h, w]
            # ([2, 3, 13, 20]) -> ([2,3])
            # from the input I take all samples, a pixel (h,w) and all his channels (:)
            input_slice.unsqueeze_(dim=2).unsqueeze_(dim=3)
            # ([2,3]) -> ([2,3,1]) -> ([2,3,1,1])
            # then I unsqueeze it to make it fit with kernel
            kernel_ = kernel[:, oc, :, :]
            # ([3, 6, 3, 3]) -> ([3, 3, 3])
            kernel_.unsqueeze_(dim=0)
            # ([3, 3, 3]) -> ([1, 3, 3, 3])
            # so now they are broadcastable
            # I take all the channels and dimensions of one kernel oc
            product = torch.sum(input=(input_slice * kernel_), dim=(1))
            # ([2, 3, 1, 1]) *
            # ([1, 3, 3, 3]) ->
            # ([2, 3, 3, 3]) -> sum over channels axis dim=1
            # ([2,    3, 3])
            # then I compress the first axis so the one of the channels by sum it
            out[
                :, oc, h * stride : h * stride + kH, w * stride : w * stride + kW
            ] += product
            # out[...] is ([2, 3, 3]) portion, insert that piece in final output with right stride, if cell was already I add the new value and sum all together
end = time.time()
print("Looping over   H, W, oC ", end - start)


start = time.time()
# try to make a faster method through broadcasting, compress #OC kernels
for h in range(H):
    for w in range(W):  # for each pixel identified by h,w
        input_slice = input[:, :, h, w]
        # ([2, 3, 13, 20]) -> ([2,3])
        # from the input I take all samples, a pixel (h,w) and all his channels (:)
        input_slice.unsqueeze_(dim=2).unsqueeze_(dim=3).unsqueeze_(dim=4)
        # ([2,3]) -> ([2,3,1]) -> ([2,3,1,1]) -> ([2,3,1,1,1])
        kernel_ = kernel.unsqueeze(dim=0)
        # ([3, 6, 3, 3]) -> ([1, 3, 6, 3, 3])
        # so now they are broadcastable
        product = torch.sum(input=(input_slice * kernel_), dim=(1))
        # ([2, 3, 1, 1, 1]) *
        # ([1, 3, 6, 3, 3]) ->
        # ([2, 3, 6, 3, 3]) -> sum over channels axis dim=1
        # ([2,  , 6, 3, 3])
        # then I compress the first axis so the one of the channels by sum it
        out[:, :, h * stride : h * stride + kH, w * stride : w * stride + kW] += product
        # out[...] is ([2, 6, 3, 3]) portion, insert that piece in final output with right stride, if cell was already I add the new value and sum all together
end = time.time()
print("Looping over , H, W,  ", end - start)
