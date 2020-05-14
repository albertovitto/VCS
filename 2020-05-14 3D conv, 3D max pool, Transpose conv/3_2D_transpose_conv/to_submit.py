import numpy as np
import torch


(n, iC, H, W) = input.shape
(iC, oC, kH, kW) = kernel.shape

pad = 0
stride = s
dilation = 1

oH = np.int((H - 1) * stride - 2 * pad + dilation * (kH - 1) + 1)
oW = np.int((W - 1) * stride - 2 * pad + dilation * (kW - 1) + 1)

out = torch.zeros(n, oC, oH, oW)
