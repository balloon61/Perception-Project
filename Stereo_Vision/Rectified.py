import numpy as np
import cv2
from utils.function import *
from matplotlib import pyplot as plt
import matplotlib

# Parameters
print("Input image you want to check (0, 1, 2):")
case = int(input())
#case = 0
if case == 0:
    img1 = cv2.imread("curule/im0.png")
    img2 = cv2.imread("curule/im1.png")
    k1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    k2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    doffs = 0
    baseline = 88.39
    width = 1920
    height = 1080
    ndisp = 220
    vmin = 55
    vmax = 195
elif case == 1:
    img1 = cv2.imread("octagon/im0.png")
   # img1 = cv2.resize(img1, (320, 240), interpolation=cv2.INTER_AREA)

    img2 = cv2.imread("octagon/im1.png")
   # img2 = cv2.resize(img2, (320, 240), interpolation=cv2.INTER_AREA)

    k1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    k2 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    doffs = 0
    baseline = 221.76
    width = 1920
#    width = 320
    height = 1080
  #  height = 240
    ndisp = 100
    vmin = 29
    vmax = 61
else:
    img1 = cv2.imread("pendulum/im0.png")
    img2 = cv2.imread("pendulum/im1.png")
    k1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    k2 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    doffs = 0
    baseline = 537.75
    width = 1920
    height = 1080
    ndisp = 180
    vmin = 25
    vmax = 150

## Step 1. Calibration ##
## Find E and F, also find R and T ##
# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# Apply ratio test
good = []
Left_points = []
Right_points = []
for m, n in matches:
    if m.distance < 0.25 * n.distance:
        good.append([m])

match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
p1, p2 = Get_Features_Array(good, kp1, kp2)
F = Estimate_F(p1, p2)
print("F:", F)

E = np.dot(np.dot(k1.T, F), k1)
u, s, v = np.linalg.svd(E)
E = np.dot(u, np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]]), v)
print("E:", E)
t, C, R, _ = decompose_R_T(E)

print("R:", R)
print("T:", C)

print("t:", t)

## Step 2. Rectification ##
## Plot Eplolar line##
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

for y, (m, n) in enumerate(matches):
    if m.distance < 0.25 * n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
print(F)
# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

cv2.imwrite("computeCorrespondEpilines1.png", img5)
cv2.imwrite("computeCorrespondEpilines2.png", img3)

_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(p1), np.float32(p2), F, imgSize=(width, height))
print(" H1: ", H1)
print(" H2: ", H2)

img1_rectified = cv2.warpPerspective(img1, H1, (width, height))
img2_rectified = cv2.warpPerspective(img2, H2, (width, height))
# Show the warp image
cv2.imwrite("Rectified_image1.png", img1_rectified)
cv2.imwrite("Rectified_image2.png", img2_rectified)

## Step 3. Disparity ##

disparity = correspondence(img1_rectified, img2_rectified, height, width, window=15, x_window=5, y_window=5)
disparity_original = disparity.copy()

d = np.max(disparity) - np.min(disparity)
for y in range(disparity.shape[0]):
    for x in range(disparity.shape[1]):
        disparity[y][x] = int((disparity[y][x] * 255) / d)


plt.figure(1)
plt.title('Disparity Grays')
plt.imshow(disparity, cmap='gray')
plt.figure(2)
plt.title('Disparity Heat')
plt.imshow(disparity, cmap='hot')

## Step 3. Depth ##

depth_image = np.zeros((height, width))
depth_value = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        depth_image[y][x] = baseline * k1[0][0] / (disparity_original[y][x] + 1)
plt.figure(3)
plt.title('Depth Map Grays')
plt.imshow(depth_image, cmap='gray')
plt.figure(4)
plt.title('Depth Map Heat')
plt.imshow(depth_image, cmap='hot')

plt.show()
















