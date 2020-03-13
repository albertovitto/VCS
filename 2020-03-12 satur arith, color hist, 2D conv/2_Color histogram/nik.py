import numpy as np
import cv2 as cv2

im = cv2.imread(
    "2020-03-12 satur arith, color hist, 2D conv/2_Color histogram/ast.png",
    cv2.IMREAD_COLOR,
)  # BGR H W 3 (400, 600, 3)
print(im.shape)
# cv2.imshow("img", im)
# cv2.waitKey()
im = im.swapaxes(0, 2)
nbin = 35
out = np.zeros((3 * nbin,))
n = im.shape[0] * im.shape[1] * im.shape[2]
for c in range(3):
    # procedure for doing the histogram on color plane c
    histogram = np.zeros((nbin,))
    for row in range(im.shape[1]):
        for col in range(im.shape[2]):
            pixel = im[c, row, col]  # scalar
            bin = pixel * nbin // 256
            histogram[bin] += 1
    a = c * nbin
    b = (c + 1) * nbin
    out[a:b] += histogram
# l1 normalization
out = out / n
