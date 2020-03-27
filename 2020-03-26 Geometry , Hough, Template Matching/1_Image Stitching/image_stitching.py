from io import BytesIO
import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import interactive

https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83
https://www.learnopencv.com/homography-examples-using-opencv-python-c/
https://www.programcreek.com/python/example/89367/cv2.findHomography
https://www.programcreek.com/python/example/89422/cv2.warpPerspective
https://github.com/spmallick/learnopencv/blob/master/Homography/homography.py
https://pylessons.com/OpenCV-image-stiching-continue/


fig = plt.figure(figsize=(30, 30))
rows = 2  # grid 2x2
columns = 2

file = open(
    "2020-03-26 Geometry , Hough, Template Matching/1_Image Stitching/gallery_0.jpg",
    mode="rb",
)
# data_files = {
#     "gallery_0.jpg" : open("gallery_0.jpg","rb").read(),
#     "gallery_1.jpg" : open("gallery_1.jpg","rb").read(),
# }
# file = BytesIO(data_files["gallery_0.jpg"])
bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)  # (27707,)
im_a = cv2.imdecode(bytes, cv2.IMREAD_COLOR)  # (287, 509, 3) BGR
im_a = np.swapaxes(
    np.swapaxes(im_a, 0, 2), 1, 2
)  # (287, 509, 3) -> (3, 287, 509) -> (3, 287, 509) BGR
im_a = im_a[::-1, :, :]  # from BGR to RGB
_, h1, w1 = im_a.shape
im_a = np.swapaxes(
    np.swapaxes(im_a, 0, 2), 1, 0
)  # (3,homograhyMatrix,W)->(W,homograhyMatrix,3)->(homograhyMatrix,W,3)

fig.add_subplot(rows, columns, 1)
plt.title("Image to stitch with the wrapped one")
plt.imshow(im_a)


# file = BytesIO(data_files["question-data/gallery_1.jpg"])
file = open(
    "2020-03-26 Geometry , Hough, Template Matching/1_Image Stitching/gallery_1.jpg",
    mode="rb",
)
bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)  # (21988,)
im_b = cv2.imdecode(bytes, cv2.IMREAD_COLOR)  # (286, 509, 3) BGR
im_b = np.swapaxes(np.swapaxes(im_b, 0, 2), 1, 2)  # (3, 286, 509) BGR
im_b = im_b[::-1, :, :]  # from BGR to RGB
c, h2, w2 = im_b.shape
im_b = np.swapaxes(
    np.swapaxes(im_b, 0, 2), 1, 0
)  # (3,homograhyMatrix,W)->(W,homograhyMatrix,3)->(homograhyMatrix,W,3)

fig.add_subplot(rows, columns, 2)
plt.title("Image to wrap")
plt.imshow(im_b)

# top left, bottom left, top right, bottom right of central painting
# then label on the left
src = np.asarray(
    [
        [138, 51],
        [130, 191],
        [338, 58],
        [332, 198],
        [76, 130],
        [95, 130],
        [95, 165],
        [76, 165],
    ],
    dtype=np.float32,
)  # im b
dst = np.asarray(
    [
        [192, 33],
        [181, 237],
        [317, 93],
        [309, 209],
        [80, 143],
        [115, 145],
        [115, 207],
        [74, 212],
    ],
    dtype=np.float32,
)  # im a

# Calculate Homography
homograhyMatrix, _ = cv2.findHomography(src, dst, cv2.RANSAC)
# # homograhyMatrix array([[ 4.05800050e+00,  1.56885289e-01, -1.42913399e+02],
#        [ 1.33026187e+00,  3.43173341e+00, -2.85531885e+02],
#        [ 8.41743350e-03,  1.02772688e-03,  1.00000000e+00]])

out = cv2.warpPerspective(
    im_b, homograhyMatrix, (w1, h1 + h2), flags=cv2.INTER_LINEAR
)  # (573, 509, 3)
out[0:h1, 0:w1] = im_a

fig.add_subplot(rows, columns, 3)
plt.title("Images stitched")
plt.imshow(out)
plt.show()
