import cv2 as cv

image = cv.imread(
    "/home/alberto/Desktop/VCS/2020-03-26 Geometry , Hough, Template Matching/4_SIFT/ast.png"
)
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d_SIFT.create()
keyPoints = sift.detect(image, None)

output = cv.drawKeypoints(image, keyPoints, None)

cv.imshow("FEATURES DETECTED", output)
cv.imshow("NORMAL", image)

cv.waitKey(0)
cv.destroyAllWindows()