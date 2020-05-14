# 3D Max Pooling

# Your code will take as input:

#     a tensor input with shape (n, iC, T, H, W);
#     a kernel temporal span kT, height kH and width kW;
#     a stride s;

# It needs then to apply a 3D max-pooling over input, using the given kernel size and the same stride s on all axes, and store the result in out. Input input has dtype torch.float32. s is an integer.

import numpy as np
import random
import torch
import torch.nn as nn

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


n = random.randint(2, 6)
iC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 5)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
stride = random.randint(2, 3)

input = torch.rand(n, iC, T, H, W)
print("in   n:{}  iC:{}   T:{}    H:{}    W:{}".format(n, iC, T, H, W))
print("ke                kT:{}   kH:{}   kW:{}".format(kT, kH, kW))
print("st   s :{}".format(stride))
oT = np.int(np.floor(1 + (T - kT) / stride))
oH = np.int(np.floor(1 + (H - kH) / stride))
oW = np.int(np.floor(1 + (W - kW) / stride))

out = torch.zeros(n, iC, oT, oH, oW)
print("ou   n:{}  iC:{}   oT:{}   oH:{}   oW:{}".format(n, iC, oT, oH, oW))
out = out.to(torch.float32)

for OT in range(oT):
    for OH in range(oH):
        for OW in range(oW):
            t_start = OT * stride
            t_end = t_start + kT
            h_start = OH * stride
            h_end = h_start + kH
            w_start = OW * stride
            w_end = w_start + kW

            this_input = input[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
            # slice of the input taking all sample and channels
            # ([5, 5, 10, 14, 18]) -> ([5, 5, 5, 5, 4])

            # max over last 3 axis so I can put the result in out
            max_pooled = torch.max(input=this_input, dim=-1)[0]  # dim=4
            # ([5, 5, 5, 5, 4]) -> ([5, 5, 5, 5])
            max_pooled = torch.max(input=max_pooled, dim=-1)[0]  # dim=3
            # ([5, 5, 5, 5]) -> ([5, 5, 5])
            max_pooled = torch.max(input=max_pooled, dim=-1)[0]  # dim=2
            # ([5, 5, 5]) -> ([5, 5])
            out[:, :, OT, OH, OW] = max_pooled
            # I put in out position OT, OH, OW a sample of size 5,5 that fits all the batch (n) and all the channels (iC)
