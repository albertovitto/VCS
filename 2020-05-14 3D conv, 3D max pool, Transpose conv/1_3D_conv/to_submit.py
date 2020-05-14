import numpy as np
import torch

(n, iC, T, H, W) = input.shape
(oC, iC, kT, kH, kW) = kernel.shape

pad = 0
stride = 1
dilation = 1

oT = np.int(np.floor(1 + (T + 2 * pad - dilation * (kT - 1) - 1) / stride))
oH = np.int(np.floor(1 + (H + 2 * pad - dilation * (kH - 1) - 1) / stride))
oW = np.int(np.floor(1 + (W + 2 * pad - dilation * (kW - 1) - 1) / stride))

out = torch.zeros(n, oC, oT, oH, oW)

for OT in range(oT):
    for OH in range(oH):
        for OW in range(oW):
            this_input = input[
                :,
                :,
                OT * stride : OT * stride + kT,
                OH * stride : OH * stride + kH,
                OW * stride : OW * stride + kW,
            ]

            this_input.unsqueeze_(dim=1)

            this_kernel = kernel.unsqueeze(dim=0)

            out[:, :, OT, OH, OW] = (
                torch.sum(input=(this_input * this_kernel), dim=(-1, -2, -3, -4)) + bias
            )  # or dim=(2,3,4,5)
