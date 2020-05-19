import numpy as np
import torch


(n, iC, H, W) = input.shape
(iC, oC, kH, kW) = kernel.shape
print("in   n:{}   iC:{}   H:{}    W:{}".format(n, iC, H, W))
print("ke   iC:{}  oC:{}  kH:{}   kW:{}".format(iC, oC, kH, kW))

pad = 0
stride = s
print("st     :{}".format(stride))
dilation = 1

oH = np.int((H - 1) * stride - 2 * pad + dilation * (kH - 1) + 1)
oW = np.int((W - 1) * stride - 2 * pad + dilation * (kW - 1) + 1)

out = torch.zeros(n, oC, oH, oW)
print("ou   n:{}  oC:{}   oH:{}   oW:{}".format(n, oC, oH, oW))

for h in range(H):
    for w in range(W):
        input_slice = input[:, :, h, w]
        input_slice.unsqueeze_(dim=2).unsqueeze_(dim=3).unsqueeze_(dim=4)
        kernel_ = kernel.unsqueeze(dim=0)
        product = torch.sum(input=(input_slice * kernel_), dim=(1))
        out[:, :, h * stride : h * stride + kH, w * stride : w * stride + kW] += product
