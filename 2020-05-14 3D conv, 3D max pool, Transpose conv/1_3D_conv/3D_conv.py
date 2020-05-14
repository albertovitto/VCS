# 3D Convolution

# Your code will take an input tensor input with shape (n, iC, T, H, W), a kernel kernel with shape (oC, iC, kT, kH, kW) and a bias bias with shape (oC, ).

# It needs then to apply a 3D convolution over input, using kernel as kernel tensor and bias as bias, using a stride of 1, no dilation, no grouping, and no padding, and store the result in out.

# input, kernel and bias are torch.Tensor with dtype torch.float32.

import numpy as np
import random
import torch
import torch.nn as nn

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 6)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, T, H, W)
print("in   n:{}  iC:{}   T:{}    H:{}    W:{}".format(n, iC, T, H, W))
kernel = torch.rand(oC, iC, kT, kH, kW)
print("ke   oc:{} iC:{}   kT:{}    kH:{}    kW:{}".format(oC, iC, kT, kH, kW))
bias = torch.rand(oC)
print("bi   oc:{}".format(oC))
pad = 0
stride = 1
dilation = 1

oT = np.int(np.floor(1 + (T + 2 * pad - dilation * (kT - 1) - 1) / stride))
oH = np.int(np.floor(1 + (H + 2 * pad - dilation * (kH - 1) - 1) / stride))
oW = np.int(np.floor(1 + (W + 2 * pad - dilation * (kW - 1) - 1) / stride))

out = torch.zeros(n, oC, oT, oH, oW)
print("ou   n:{}  oC:{}   oT:{}   oH:{}   oW:{}".format(n, oC, oT, oH, oW))
out = out.to(torch.float32)

for OT in range(oT):
    for OH in range(oH):
        for OW in range(oW):
            # input n:5  iC:5   T:14    H:18    W:17
            # input[:, :, time:time+kT, row:row+kH, col:col+kW]
            this_input = input[
                :,
                :,
                OT * stride : OT * stride + kT,
                OH * stride : OH * stride + kH,
                OW * stride : OW * stride + kW,
            ]
            # ([5, 5, 14, 18, 17]) -> ([5, 5, 5, 4, 5])

            this_input.unsqueeze_(dim=1)
            # ([5, 5, 5, 4, 5]) -> ([5, 1, 5, 5, 4, 5])

            this_kernel = kernel.unsqueeze(dim=0)
            # ([2, 5, 5, 4, 5]) -> ([1, 2, 5, 5, 4, 5])

            # so now they are broadcastable and can do element wise multiplication, then sum over last 4 axis so it remains a 5,2 tensor to put in the final one

            # ([5, 1, 5, 5, 4, 5])
            # ([1, 2, 5, 5, 4, 5])
            #   0, 1, 2, 3, 4, 5
            # sum over axis (2,3,4,5) -> -1, -1, -1, -1
            #   5, 2, 1, 1, 1, 1

            out[:, :, OT, OH, OW] = (
                torch.sum(input=(this_input * this_kernel), dim=(-1, -2, -3, -4)) + bias
            )  # or dim=(2,3,4,5)


m = nn.Conv3d(
    in_channels=iC,
    out_channels=oC,
    kernel_size=(kT, kH, kW),
    bias=True,
    stride=stride,
    padding=pad,
    dilation=dilation,
)
# out_pytorch = m(input)
# print(torch.all(out.eq(out_pytorch)))
