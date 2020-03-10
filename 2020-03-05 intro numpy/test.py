import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable

x = torch.randn(2, 2)
y = torch.Tensor([1])
z = Variable(torch.ones(2, 3))

print(x, y, z)


x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()
