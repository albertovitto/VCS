import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt
import random

# Your code will take as input a mini-batch of feature maps input (a np.ndarray tensor with dtype np.float32 and shape (n, H, W)), and a template template (a np.ndarray with dtype np.float32 and shape (kH, kW)). It then needs to compare the template against all samples in the mini-batch in a sliding window fashion, and store the result in out.

# out will have shape (n, oH, oW), where oH=iH-(kH-1) and oW=iW-(kW-1), and out[i, :, :] will contain the similarity between the template and the i-th feature map at all valid locations. Use the sum of squared differences as comparison function.

random.seed(322)

n = random.randint(1, 3)
H = random.randint(10, 20)
W = random.randint(10, 20)
input = np.random.rand(n, H, W).astype(np.float32)  # (3, 16, 12)
n, iH, iW = input.shape
print("input ", input.shape)

kH = random.randint(2, 6)
kW = random.randint(2, 6)
template = np.random.rand(kH, kW).astype(np.float32)  # (2, 3)
kH, kW = template.shape
print("template ", template.shape)

oH = iH - (kH - 1)
oW = iW - (kW - 1)
out = np.random.rand(n, oH, oW).astype(np.float32)  # (3, 15, 10)
print("out ", out.shape)

template = np.expand_dims(
    template, axis=0
)  # (2, 3) -> (1, 2, 3) so i can broadcast it with input slice

for OH in range(oH):
    for OW in range(oW):
        h_start = OH  # vertical
        h_end = OH + kH
        w_start = OW  # horizontal
        w_end = OW + kW
        input_slice = input[:, h_start:h_end, w_start:w_end]
        # input (n, ih, iw) -> (n, 2, 3)
        # template             (1, 2, 3)
        # =
        # out                  (n, 2, 3) posso usare broadcast
        # axis=(-1, -2) cioè fai somma per tutte le feature map (N) su quella porzione 2x3 del template e dell input
        # (-1, -2) significa ultima e penultima dimensione, uguale a scrivere (1,2)
        out[:, OH, OW] = np.sum(((input_slice - template) ** 2), axis=(-1, -2))
