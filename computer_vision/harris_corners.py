import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

filename = '../data/left01.jpg'
img = cv.imread(filename, 1)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# find Harris corners
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
dst = cv.dilate(dst,None)
ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
# find centroids
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
# define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# Now draw them
for c in centroids:
    cv.circle(img, c.astype(int),1, (0, 0, 255), 2 )

for c in corners:
    cv.circle(img, c.astype(int),1, (0, 255,0), 2 )

cv.imshow('out',img)
cv.waitKey(0)
cv.imwrite('subpixel5.png',img)