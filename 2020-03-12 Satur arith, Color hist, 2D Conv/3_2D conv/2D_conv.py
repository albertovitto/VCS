# Your code will take an input tensor input with shape (n, iC, H, W) and a kernel kernel with shape (oC, iC, kH, kW). It needs then to apply a 2D convolution over input, using kernel as kernel tensor, using a stride of 1, no dilation, no grouping, and no padding, and store the result in out. Both input and kernel have dtype np.float32.

# http://www.cs.cmu.edu/~aharley/vis/conv/flat.html
# https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


random.seed(1)
print("input")
n = random.randint(2, 6)  # numero di sample
iC = random.randint(2, 6)  # canali immagine, es rgb =3
H = random.randint(10, 20)  # height
W = random.randint(10, 20)  # width
print("n  ", n, " iC  ", iC, " H  ", H, " W  ", W)
input = np.random.rand(n, iC, H, W)
print(input.shape)
# input
# n   3  iC   6  H   11  W   14
# (3, 6, 11, 14)

print("\nkernel")
oC = random.randint(2, 6)  # numero di filtri = numero canali uscita
kH = random.randint(2, 6)  # height di kernel
kW = random.randint(2, 6)  # width di kernel
print("oC  ", oC, " iC  ", iC, " kH  ", kH, " kW  ", kW)
kernel = np.random.rand(oC, iC, kH, kW)
print(kernel.shape)
# kernel
# oC   2  iC   6  kH   5  kW   5
# (2, 6, 5, 5)

# calcolo dimensioni di array dopo convoluzione di tutti i sample
pad = 0
stride = 1
dilation = 1
oH = ((H + (2 * pad) - (dilation * (kH - 1)) - 1) / stride) + 1
oW = ((W + (2 * pad) - (dilation * (kW - 1)) - 1) / stride) + 1
out1 = np.zeros(
    shape=(int(np.floor(n)), int(np.floor(oC)), int(np.floor(oH)), int(np.floor(oW))),
)  # floor arrotonda a intero inferiore, int me lo rende int
out2 = np.zeros(
    shape=(int(np.floor(n)), int(np.floor(oC)), int(np.floor(oH)), int(np.floor(oW))),
)
print("\nout")
print("n  ", n, " oC  ", oC, " oH  ", oH, " oW  ", oW)
print(out1.shape)
# out
# n   3  oC   2  oH   7.0  oW   10.0
# (3, 2, 7, 10)

# convoluzione applicata a un blocco di immagine 3d con il kernel 3d che ci
# scorre dentro in altezza e larghezza, NO in depth
def twoD_conv_single_step(input_slice, kernel_weights, bias=0):
    s = np.multiply(input_slice, kernel_weights)  # element wise moltiplicazione
    # è un array 3D, devono avere la stessa dimensione! es shape (6, 5, 5) (6, 5, 5)
    # Z = np.sum(s)
    Z = s.sum()  # sommo tutti i suoi valori
    # Z = Z + bias.astype(float)

    return Z


# convoluzione applicata a tutto il batch di immagini con tutti i kernel
def twoD_conv_forward(input, kernel, output, pad, stride, dilation, bias=0):
    n, iC, H, W = input.shape
    oC, _, kH, kW = kernel.shape
    _, _, oH, oW = output.shape
    for N in range(n):  # scorro su ogni sample
        input_sample = input[
            N, :, :, :
        ]  # ne prendo 1, da 4d a 3d(canali,altezza, larghezza)
        for OH in range(oH):
            # scorro su posizioni di height in array uscita
            count = 0  # per debug
            for OW in range(oW):
                # scorro su posizioni di height in array uscita
                for OC in range(oC):
                    # scorro sui canali che avrà sample uscita dopo tutte le convoluzioni su tutti i canali con tutti i kernel
                    vertical_start = OH * stride
                    # calcolo dimensioni di maschera che dovrà scorrere sopra ogni matrice 3d con i canali
                    vertical_end = OH * stride + kH
                    horizontal_start = OW * stride
                    horizontal_end = OW * stride + kW

                    # input_slice è pezzo della matrice sample attuale: da essa estraggo tutti i canali (:) e solo un pezzo in altezza e larghezza in base a slicing (estraggo pezzi di matrice rettangolari o quadrate), poi mi sposterò in base allo stride a dx e poi in bassp
                    input_slice = input_sample[
                        :, vertical_start:vertical_end, horizontal_start:horizontal_end
                    ]
                    print(count, OC, OH, OW)
                    print(input_slice.shape, kernel[OC, :, :, :].shape)
                    # vado a riempire l'input in posizione N,OC,OH,OW
                    # del kernel passo la matrice 3d relativa al kernel attuale OC (numero di kernel in uso)
                    output[N, OC, OH, OW] = twoD_conv_single_step(
                        input_slice, kernel[OC, :, :, :], bias=0
                    )
                    count = count + 1
    return output


def twoD_conv_forward_faster(input, kernel, output, pad, stride, dilation, bias=0):
    n, iC, H, W = input.shape
    oC, _, kH, kW = kernel.shape
    _, _, oH, oW = output.shape
    for OH in range(oH):
        for OW in range(oW):
            # input[:, :, row:row+kH, col:col+kW] ==> (n,  iC, kH, kW)
            # kernel ==>                              (oC, iC, kH, kW)

            this_input = np.expand_dims(input[:, :, OH : OH + kH, OW : OW + kW], 1)
            # (n, 1,  iC, kH, kW)
            this_kernel = np.expand_dims(kernel, 0)
            # (1, oC, iC, kH, kW)

            # this_input * this_kernel
            #     (n, 1,  iC, kH, kW)
            #     (1, oC, iC, kH, kW)
            # ==> (n, oC, iC, kH, kW)

            # np.sum(tensor) = np.sum(tensor, axis=[all axes]) ==> scalar

            # Summation over the last three axes
            # np.sum(np.sum(np.sum(.., axis=-1), axis=-1), axis=-1)
            #         axis : None or int or tuple of ints, optional

            # Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis.

            # New in version 1.7.0.

            # If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single axis or all the axes as before.

            # np.sum(.., axis=(-1, -2, -3))
            # ==> (n, oC)

            output[:, :, OH, OW] = np.sum(this_input * this_kernel, axis=(-1, -2, -3))
            # out[:, :, row, col] ==> (n, oC)
    return output


out1 = twoD_conv_forward(input, kernel, out1, pad, stride, dilation, bias=0)
out2 = twoD_conv_forward_faster(input, kernel, out2, pad, stride, dilation, bias=0)
print(np.array_equal(out1, out2))
# np.random.seed(1)
# a_slice_prev = np.random.randn(iC, H, W)
# W = np.random.randn(iC, kH, kW)
# b = np.random.randn(1, 1, 1)

# Z = twoD_conv_single_step(a_slice_prev, W, b)
# print("Z =", Z)

