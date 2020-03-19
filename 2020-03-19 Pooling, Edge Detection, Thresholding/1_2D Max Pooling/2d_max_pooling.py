# 2D Max Pooling

# Your code will take as input:

#     a tensor input with shape (n, iC, H, W);
#     a kernel height kH and a kernel width kW;
#     a stride s;

# It needs then to apply a 2D max-pooling over input, using the given kernel size and stride, and store the result in out. Input input has dtype np.float32.

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from skimage.measure import block_reduce
import random

random.seed(1)
n = random.randint(2, 6)  # numero di sample
iC = random.randint(2, 6)  # canali immagine, es rgb =3
H = random.randint(10, 20)  # height
W = random.randint(10, 20)  # width
kH = random.randint(2, 5)  # kernel height
kW = random.randint(2, 5)  # kernel width
s = random.randint(1, 2)
input = np.random.rand(n, iC, H, W)
print("input")
print("n  ", n, " iC  ", iC, " H  ", H, " W  ", W)
print(input.shape)
print("kH  ", kH, " kW  ", kW, " stride  ", s)


def twoD_max_pooling(input, kH, kW, pad, stride, dilation):
    n, iC, H, W = input.shape
    oH = ((H + (2 * pad) - (dilation * (kH - 1)) - 1) / stride) + 1
    oH = int(np.floor(oH))
    oW = ((W + (2 * pad) - (dilation * (kW - 1)) - 1) / stride) + 1
    oW = int(np.floor(oW))
    output = np.zeros(shape=(n, iC, oH, oW), dtype=np.float32)
    print("\noutput")
    print("n  ", n, " iC  ", iC, " oH  ", oH, " oW  ", oW)
    print(output.shape)

    # n  ic  (h  w)  input
    # 1  1   (kh kw) kernel

    # devo lavorare sul rettangolo h*w, perchè i canali ic li faccio separatamente
    # così come il numero di sample n
    # quindi lavoro sugli ultimi due assi h e w

    # n  ic  oh ok   output

    for OH in range(oH):
        for OW in range(oW):
            vertical_start = OH * stride
            vertical_end = OH * stride + kH
            horizontal_start = OW * stride
            horizontal_end = OW * stride + kW

            this_input = input[
                :, :, vertical_start:vertical_end, horizontal_start:horizontal_end
            ]
            output[:, :, OH, OW] = np.amax(this_input, axis=(-1, -2))
    print("ok")
    # for N in range(n):  # scorro su ogni sample
    #     input_sample = input[
    #         N, :, :, :
    #     ]  # ne prendo 1, da 4d a 3d(canali,altezza, larghezza)
    #     for OH in range(oH):
    #         # scorro su posizioni di height in array uscita
    #         count = 0  # per debug
    #         for OW in range(oW):
    #             # scorro su posizioni di height in array uscita
    #             for OC in range(oC):
    #                 # scorro sui canali che avrà sample uscita dopo tutte le convoluzioni su tutti i canali con tutti i kernel
    #                 vertical_start = OH * stride
    #                 # calcolo dimensioni di maschera che dovrà scorrere sopra ogni matrice 3d con i canali
    #                 vertical_end = OH * stride + kH
    #                 horizontal_start = OW * stride
    #                 horizontal_end = OW * stride + kW

    #                 # input_slice è pezzo della matrice sample attuale: da essa estraggo tutti i canali (:) e solo un pezzo in altezza e larghezza in base a slicing (estraggo pezzi di matrice rettangolari o quadrate), poi mi sposterò in base allo stride a dx e poi in bassp
    #                 input_slice = input_sample[
    #                     :, vertical_start:vertical_end, horizontal_start:horizontal_end
    #                 ]
    #                 print(count, OC, OH, OW)
    #                 print(input_slice.shape, kernel[OC, :, :, :].shape)
    #                 # vado a riempire l'input in posizione N,OC,OH,OW
    #                 # del kernel passo la matrice 3d relativa al kernel attuale OC (numero di kernel in uso)
    #                 output[N, OC, OH, OW] = twoD_conv_single_step(
    #                     input_slice, kernel[OC, :, :, :], bias=0
    #                 )
    #                 count = count + 1
    return output


# a = block_reduce(input, (kH, kW), np.max)

pad = 0
dilation = 1
out = twoD_max_pooling(input, kH, kW, pad, s, dilation)
