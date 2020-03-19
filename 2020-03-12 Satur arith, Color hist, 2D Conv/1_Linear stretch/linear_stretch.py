import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib

# image = data.coffee()
# io.imsave("cof.png", image)
im = cv2.imread(
    "2020-03-12 satur arith, color hist, 2D conv/1_Linear stretch/cof.png",
    cv2.IMREAD_COLOR,
)  # BGR H W 3 (400, 600, 3)
# cv2.imshow("img", im)
# cv2.waitKey()
# print(im.shape)
# print("image dtype ", img.dtype)
im = im.swapaxes(0, 2)  # BGR 3 W H (3, 600, 400)
print(im.shape, im.dtype)

# oppure
# plt.imshow(img[::-1].swapaxes(2, 0))  # BGR -> RGB H W 3
# plt.show()

# img = np.arange(12).reshape(2, 2, 3)  #  base 2x3, altezza 2
# print(img)
# print("\n")
# print(img[::-1], "\n", img[::-1].shape)
# print("\n")
# print(img[::-1].swapaxes(2, 0), img[::-1].swapaxes(2, 0).shape)

a = 1.8572174313946244

b = -34.29745793809741

out_float = im.astype(np.float32)  # np.empty_like(im,dtype=np.float)
print(out_float.shape, out_float.dtype)
# print(out_float[:, 0, 0])
# print(out_float.shape[1], out_float.shape[2], out_float[0, 0, 0])
for w in range(out_float.shape[1]):  # rows
    for h in range(out_float.shape[2]):  # cols
        out_float[:, w, h] = a * out_float[:, w, h] + b  # r,g,b with broadcast

# out_float = out_float * a + b

print("ok")
out = np.empty_like(im)
out_float = out_float.round()
print("ok")
np.clip(out_float, 0, 255, out=out)
print("ok")

cv2.imshow("out", out.swapaxes(2, 0))
cv2.waitKey()
# plt.imshow(out[::-1].swapaxes(2, 0))  # BGR -> RGB H W 3
# plt.show()

# simpler
# out = np.empty_like(im)
# t=im*a+b
# np.clip(np.round(t),0,255,out)
