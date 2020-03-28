import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt

# https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

fig = plt.figure(figsize=(30, 30))
rows = 2  # grid 2x2
columns = 2

im = data.coins()  # [160:230, 70:270]  # (70, 200)
if len(im.shape) == 3:  # converting to gray scale if image is RGB
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
h, w = im.shape

fig.add_subplot(rows, columns, 1)
plt.title("Original gray")
plt.imshow(im, cmap="gray")

# Blur the image to reduce noise
img_blur = cv2.medianBlur(im, 5)
fig.add_subplot(rows, columns, 2)
plt.title("Blurred")
plt.imshow(im, cmap="gray")

# Apply hough transform on the image
circles = cv2.HoughCircles(
    image=img_blur,
    method=cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=15,
    param1=200,
    param2=10,
    minRadius=20,
    maxRadius=30,
)
# Draw detected circles
if circles is not None:
    circles = np.uint16(
        np.around(circles)
    )  ## convert the (x, y) coordinates and radius of the circles to integers
    for (x, y, r) in circles[0, :]:
        # Draw outer circle
        cv2.circle(img=im, center=(x, y), radius=r, color=(255, 255, 0), thickness=2)
        # Draw inner circle
        cv2.circle(img=im, center=(x, y), radius=1, color=(0, 0, 0), thickness=2)
fig.add_subplot(rows, columns, 3)
plt.title("With Hough")
plt.imshow(im)  # cmap="gray"

plt.show()

# img = frame_a.copy()
# img = cv2.resize(img, (0,0), fx=.25, fy=.25)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# # Call the edge detector
# edges = cv2.Canny(gray, 50, 500,apertureSize=3)

# # Detect lines via HoughLinesP
# minLineLength = 60
# maxLineGap = 1
# lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)

# # Display the result
# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# plt.imshow(img[:,:,::-1])
