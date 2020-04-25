import numpy as np
import random
import torch


(N, C, H, W) = input.shape
(L, _) = boxes[0].shape
(oH, oW) = output_size[0], output_size[1]

out = torch.zeros(N, L, C, oH, oW)
out = out.to(torch.float32)


def get_indexes(i, j, y1, x1, y2, x2, oH, oW):
    out = []
    y_start = torch.floor(y1 + i * (y2 - y1 + 1) / oH)
    out.append(y_start)
    y_end = torch.ceil(y1 + (i + 1) * (y2 - y1 + 1) / oH)
    out.append(y_end)
    x_start = torch.floor(x1 + j * (x2 - x1 + 1) / oW)
    out.append(x_start)
    x_end = torch.ceil(x1 + (j + 1) * (x2 - x1 + 1) / oW)
    out.append(x_end)
    out = [x.type(torch.int32) for x in out]
    return out


for n in range(N):
    for l in range(L):
        for i in range(oH):
            for j in range(oW):
                (y1, x1, y2, x2) = boxes[n][l]
                (y1, x1, y2, x2) = torch.round(boxes[n][l])
                (y_start, y_end, x_start, x_end) = get_indexes(
                    i, j, y1, x1, y2, x2, oH, oW
                )
                slice = input[n, :, y_start:y_end, x_start:x_end]
                slice, _ = torch.max(torch.max(slice, dim=1)[0], dim=1)
                out[n, l, :, i, j] = slice
