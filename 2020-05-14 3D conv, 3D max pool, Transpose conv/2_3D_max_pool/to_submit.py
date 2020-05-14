import numpy as np
import random
import torch
import torch.nn as nn

(n, iC, T, H, W) = input.shape

oT = np.int(np.floor(1 + (T - kT) / s))
oH = np.int(np.floor(1 + (H - kH) / s))
oW = np.int(np.floor(1 + (W - kW) / s))

out = torch.zeros(n, iC, oT, oH, oW)

for OT in range(oT):
    for OH in range(oH):
        for OW in range(oW):
            t_start = OT * s
            t_end = t_start + kT
            h_start = OH * s
            h_end = h_start + kH
            w_start = OW * s
            w_end = w_start + kW

            this_input = input[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
            max_pooled = torch.max(input=this_input, dim=-1)[0]
            max_pooled = torch.max(input=max_pooled, dim=-1)[0]
            max_pooled = torch.max(input=max_pooled, dim=-1)[0]
            out[:, :, OT, OH, OW] = max_pooled
