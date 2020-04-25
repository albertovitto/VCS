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
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://medium.com/@erikgaas/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem-145620f93096
# https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624
# https://github.com/yunjey/pytorch-tutorial/blob/57afe85b2c7e6bb92918a73b9a9b6a3394c92951/tutorials/02-intermediate/deep_residual_network/main.py#L54
# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if (stride != 1) or (in_channels != out_channels):
            residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        out += residual
        out = self.relu(out)
        return out
