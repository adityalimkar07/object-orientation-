import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

def detect_and_match_feature(image1_path, image2_path):
    img1 = cv.imread(image1_path)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    # Detect Harris corners
    gray1 = np.float32(gray1)
    dst = cv.cornerHarris(gray1, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(np.float32(gray1), np.float32(centroids), (5, 5), (-1, -1), criteria)

    prominent_corner = corners[1]  
    x, y = int(prominent_corner[0]), int(prominent_corner[1])
    template_size = 40  # Define the size of the template patch
    top_left_x = max(0, x - template_size // 2)
    top_left_y = max(0, y - template_size // 2)
    bottom_right_x = min(gray1.shape[1], x + template_size // 2)
    bottom_right_y = min(gray1.shape[0], y + template_size // 2)
    template = gray1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    template = np.uint8(template)

    img1_with_feature = img1.copy()
    cv.circle(img1_with_feature, (x, y), template_size // 2, (0, 255, 0), 2)

    img2 = cv.imread(image2_path)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv.matchTemplate(gray2, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    img2_with_feature = img2.copy()
    cv.rectangle(img2_with_feature, top_left, bottom_right, (0, 255, 0), 2)

    plt.figure(figsize=(12, 6))
    plt.subplot(131), plt.imshow(cv.cvtColor(img1_with_feature, cv.COLOR_BGR2RGB)), plt.title('Image 1 with Feature')
    plt.subplot(132), plt.imshow(template, cmap='gray'), plt.title('Template Extracted') #comment to remove template plot
    plt.subplot(133), plt.imshow(cv.cvtColor(img2_with_feature, cv.COLOR_BGR2RGB)), plt.title('Detected Feature in Image 2')
    plt.show()
    # Save the result images if interested
'''    cv.imwrite('cropped_testimage.jpg', img1_with_feature)
    cv.imwrite('cropped_templateimage.jpg', img2_with_feature)'''

image1_path = 'cropped_templateimage.jpg'
image2_path = 'cropped_testimage.jpg'

# Run the detection and matching
detect_and_match_feature(image1_path, image2_path)
    # l = [x,y,top_left[0]/2 + bottom_right[0]/2,top_left[1]/2 + bottom_right[1]] 
    # al = l[0],l[1]
    # bl = l[2],l[3]

#Calcultating angular displacement still remains
'''ac = [450,450]  # assuming object detection is accurate and thus com_frame x,y = 450,450
bc = [450,450]
bl = []
ax = int(al[0] - ac[0])
ay = int(al[1] - ac[1])
bx = int(bl[0] - bc[0])
by = int(bl[1] - bc[1])
sum = ax*bx + ay*by
magn = ((ax**2 + ay**2)*(bx**2 + by**2))**0.5
dotp = sum/magn
print(math.degrees(math.acos(dotp)))'''