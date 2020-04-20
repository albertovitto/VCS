import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("OpenCV-Python Tutorials/messi5.jpg", 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(231), plt.imshow(img, cmap="gray")
plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])


def floor_and_int(n):
    return int(np.floor(n))


rows, cols = img.shape
crow, ccol = floor_and_int(rows / 2), floor_and_int(cols / 2)
fshift[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)


plt.subplot(234), plt.imshow(img_back, cmap="gray")
plt.title("Image after HPF"), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(img_back)
plt.title("Result in JET"), plt.xticks([]), plt.yticks([])


plt.show()
