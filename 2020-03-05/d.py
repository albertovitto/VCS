import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

x = torch.randn(2, 2)
y = torch.Tensor([1])
z = Variable(torch.ones(2, 3))

print(x, y, z)
