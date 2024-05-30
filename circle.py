import cv2
import numpy as np
import math

def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    edges = cv2.Canny(gray_blurred, 50, 150)
    return edges

def detect_circles(image, min_radius=0, max_radius=0):
    edges = preprocess_image(image)
    

    circles = cv2.HoughCircles(edges, 
                               cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, 
                               param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius)

    return circles

def filter_circles(image, circles):
    filtered_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            mask = np.zeros_like(image)
            cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
            masked_image = cv2.bitwise_and(image, mask)
            gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.8 <= circularity <= 1.2:  # Adjust circularity threshold as needed
                    filtered_circles.append(circle)
    return filtered_circles

def find_key_circles(image, circles):
    if circles is None:
        return None, None

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    def distance_from_center(circle):
        x, y, r = circle
        return np.sqrt((x - center[0])**2 + (y - center[1])**2)

    center_circle = min(circles, key=distance_from_center)

    largest_farthest_circle = max(circles, key=lambda c: (distance_from_center(c), c[2]))

    return center_circle, largest_farthest_circle

def draw_circles(image, circles):
    for circle in circles:
        x, y, r = circle
        print(f'Circle center: ({x}, {y}), Radius: {r}')

        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

# Load the cropped template image
image = cv2.imread('cropped_templateimage.jpg')

min_radius = 55  # Adjust based on expected circle size
max_radius = 100  # Adjust based on expected circle size
circles = detect_circles(image, min_radius, max_radius)

filtered_circles = filter_circles(image, circles)

# Find the key circles
center_circle, largest_farthest_circle = find_key_circles(image, filtered_circles)
ac = center_circle
al = largest_farthest_circle
# Draw the detected circles
if center_circle is not None and largest_farthest_circle is not None:
    draw_circles(image, [center_circle, largest_farthest_circle])

# Show the result
cv2.imshow('Detected Circle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the cropped test image
image = cv2.imread('cropped_testimage.jpg')

min_radius = 55  # Adjust based on expected circle size
max_radius = 100  # Adjust based on expected circle size
circles = detect_circles(image, min_radius, max_radius)

filtered_circles = filter_circles(image, circles)

# Find the key circles
center_circle, largest_farthest_circle = find_key_circles(image, filtered_circles)
bc = center_circle
bl = largest_farthest_circle
# Draw the detected circles
if center_circle is not None and largest_farthest_circle is not None:
    draw_circles(image, [center_circle, largest_farthest_circle])

# Show the result
cv2.imshow('Detected Circle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

ax = int(al[0] - ac[0])
ay = int(al[1] - ac[1])
bx = int(bl[0] - bc[0])
by = int(bl[1] - bc[1])
sum = ax*bx + ay*by
magn = ((ax**2 + ay**2)*(bx**2 + by**2))**0.5
dotp = sum/magn
print(math.degrees(math.acos(dotp)))
