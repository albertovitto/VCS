import numpy as np
import random
import torch
import torch.nn as nn

x = (
    torch.tensor([[0, 1], [2, 3]], dtype=torch.long).reshape(1, 1, 2, 2).long()
)  # type(torch.LongTensor)
x = x.float()
k = (
    torch.tensor([[0, 1], [2, 3]], dtype=torch.long).reshape(1, 1, 2, 2).float()
)  # type(torch.LongTensor)
# With square kernels and equal stride
m = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=1)
m.weight = nn.Parameter(k, requires_grad=True)
output = m(x)
# exact output size can be also specified as an argument
input = torch.randn(1, 16, 12, 12)
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
h.size()
torch.Size([1, 16, 6, 6])
output = upsample(h, output_size=input.size())
output.size()
torch.Size([1, 16, 12, 12])
