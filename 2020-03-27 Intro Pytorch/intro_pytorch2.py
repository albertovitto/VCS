import torch
import numpy as np
import time

np.random.seed(1)

x = torch.Tensor(5, 3)
print(x)

x = torch.rand(5, 3)
print(x, x.size(), x.shape)
print(x[:, 1])

y = torch.rand(5, 3)
print(x + y)

# Numpy bridge
a = torch.ones(size=[5])
b = a.numpy()
print(a, b)

a.mul_(2)
print(a, b)

a = np.ones(shape=(5, 1))
b = torch.from_numpy(a)
print(a, b)
np.multiply(a, 2, out=a)
print(a, b)
