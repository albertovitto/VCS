# Your code will take an input tensor input with shape (n, iC, H, W), a kernel kernel with shape (iC, oC, kH, kW) and a stride s.

# It needs then to apply a 2D Transpose convolution over input, using kernel as kernel tensor, using a stride of s on both axes, no dilation, no grouping, and no padding, and store the result in out.

# input and kernel are torch.Tensor with dtype torch.float32. s is an integer.

import numpy as np
import random
import torch

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, H, W)
kernel = torch.rand(iC, oC, kH, kW)
print("in   n:{}   iC:{}   H:{}    W:{}".format(n, iC, H, W))
print("ke   iC:{}  oC:{}  kH:{}   kW:{}".format(iC, oC, kH, kW))

pad = 0
stride = random.randint(2, 6)
dilation = 1

oH = np.int((H - 1) * stride - 2 * pad + dilation * (kH - 1) + 1)
oW = np.int((W - 1) * stride - 2 * pad + dilation * (kW - 1) + 1)

out = torch.zeros(n, oC, oH, oW)
print("ou   n:{}  oC:{}   oH:{}   oW:{}".format(n, oC, oH, oW))
out = out.to(torch.float32)
