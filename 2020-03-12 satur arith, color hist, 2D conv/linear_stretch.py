# """ # Your code will
# take as input a color image im (a np.ndarray with dtype np.uint8 and rank 3)
#  and two scalars a and b. It must apply a pixel-wise linear transformation
#  (every pixel p is transformed to a⋅p+b).
#  The code should produce a new image out with the same shape and dtype.

# # a and b can be either ints or floats. Be careful to: compute the exact result,
# # round to nearest integer and then clip between 0 and 255.
#  """

import numpy as np
import cv2 as cv2
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib

image = data.astronaut()
io.imsave("ast.png", image)
print(image.shape)
plt.imshow(image)
plt.show()

