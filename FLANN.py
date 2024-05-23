import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

img1 = cv2.imread('cropped_templateimage.jpg', 0) 
img2 = cv2.imread('cropped_testimage.jpg', 0)  

sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.12)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) 
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance: # lesser the value better the goodmatch
        good_matches.append(m)

matchesMask = [0] * len(good_matches)  
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

best_match = good_matches[0]
(x1, y1) = kp1[best_match.queryIdx].pt
(x2, y2) = kp2[best_match.trainIdx].pt

cv2.circle(img1, (int(x1), int(y1)), 10, (0, 255, 0), 2)
cv2.circle(img2, (int(x2), int(y2)), 10, (0, 255, 0), 2)

com_img1 = np.array([x1, y1])
com_img2 = np.array([x2, y2])

plt.subplot(121), plt.imshow(img1, cmap='gray'), plt.title('Image 1')
plt.subplot(122), plt.imshow(img2, cmap='gray'), plt.title('Image 2')
plt.show()

print("Center of circle of the matched feature in Image 1:", com_img1)
print("Center of circle of the matched feature in Image 2:", com_img2)

ac = [450,450]  # assuming object detection is accurate and thus com_frame x,y = 450,450
bc = [450,450]
al = com_img1
bl = com_img2
ax = int(al[0] - ac[0])
ay = int(al[1] - ac[1])
bx = int(bl[0] - bc[0])
by = int(bl[1] - bc[1])
sum = ax*bx + ay*by
magn = ((ax**2 + ay**2)*(bx**2 + by**2))**0.5
dotp = sum/magn
print(math.degrees(math.acos(dotp)))