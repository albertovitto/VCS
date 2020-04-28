# Residual block

# A residual block is defined as
# y=σ(F(x)+G(x))
# where x and y represent the input and output tensors of the block, σ is the ReLU activation function, F is the residual function to be learned and G is a projection shortcut used to match dimensions between F(x) and x

# Your code needs to define a ResidualBlock class (inherited from nn.Module) which implements a residual block. In your code, F
# will be implemented with two convolutional layers with a ReLU non-linearity between them, i.e. F=conv2(σ(conv1(x)))

# . Batch normalization will also be adopted right after each convolution operation.

# The constructor of the ResidualBlock class needs to take the following arguments as input:

# inplanes, the number of channels of x
# planes, the number of output channels of conv1 and conv2
# stride, the stride of conv1


# If the shapes of F(x)
# and x do not match (either because inplanes != planes, or because stride > 1) ResidualBlock also needs to apply a projection shortcut G, composed of a convolutional layer with kernel size 1×1, no bias, no padding and stride stride, followed by a batch normalization.

# The forward method of ResidualBlock will take as input the input tensor x
# and return the output tensor y, after performing all the operations of a Residual block.

# Additional details

# Unless otherwise specified, convolutional layers must have 3×3
# kernels, stride 1, padding 1 and no bias.
# Documentation of nn.Module

import numpy as np
import torch
import torch.nn as nn
import random

np.random.seed(0)
torch.manual_seed(0)


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride, self.inplanes, self.planes = stride, inplanes, planes
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if (self.stride > 1) or (self.inplanes != self.planes):
            residual = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    self.planes,
                    kernel_size=1,
                    stride=self.stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.planes),
            )
        y += residual
        y = self.relu(y)
        return y


n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
print("n  ", n, " iC  ", iC, " H  ", H, " W  ", W)
input = torch.rand(n, iC, H, W)
oC = random.randint(2, 6)
print("oC  ", oC)
r = ResidualBlock(inplanes=(iC), planes=(oC), stride=1)
r.forward(input)
a = 0
