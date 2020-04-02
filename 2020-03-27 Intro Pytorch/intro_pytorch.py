import torch
import numpy as np
import time

np.random.seed(1)

d = 3000
a = np.random.rand(d, d)
b = np.random.rand(d, d)
start = time.time()
c = np.matmul(a, b)
end = time.time()
print(
    "Elapsed time to do {}*{} x {}*{} multiplication with Numpy: {} seconds".format(
        d, d, d, d, end - start
    )
)

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.rand(size=(d, d), device=gpu)
b = torch.rand(size=(d, d), device=gpu)
start = time.time()
c = torch.mm(a, b)
end = time.time()
print(
    "Elapsed time to do {}*{} x {}*{} multiplication with Pytorch: {} seconds".format(
        d, d, d, d, end - start
    )
)
