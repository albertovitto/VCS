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
s = random.randint(1, 2)  # stride
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

            # prendo pezzo di input che comprende la dimensione
            # data dal kernel; in piu prendo tutti i canali di tutti i sample
            this_input = input[
                :, :, vertical_start:vertical_end, horizontal_start:horizontal_end
            ]
            # calcolo quindi su ogni finestra di dimensione kH,kW il valore max
            # essendo this_input un array 4D (n,iC,kH,kW) a me interessano
            # le ultime due dimensioni su cui fare il max, queste dimensioni
            # le seleziono con axis=(-1, -2) cioè prendi ultima (kW) e prendi penulitma (kH); era indifferente scrivere axis=(2,3) cioè prendi assi in posizione 2 e 3 (n,iC,kH,kW) -> (0,1,2,3)
            output[:, :, OH, OW] = np.amax(this_input, axis=(-1, -2))
            # così facendo risparmio due for loop su n e iC e sfrutto il broadcasting di numpy; faccio contemporaneamente tutti i canali di tutte le immagini del punto di output OH,OW

    return output


pad = 0
dilation = 1
out = twoD_max_pooling(input, kH, kW, pad, s, dilation)
