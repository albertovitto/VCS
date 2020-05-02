import torch
import torch.nn as nn


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
            residual = torch.Tensor(residual)
        y += residual
        y = self.relu(y)
        return y
