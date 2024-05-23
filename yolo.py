#Step1 getting centre coordinates of detected objects in both test and template
from ultralytics import YOLO
import os
import pandas as pd
# Load YOLO model
image_filename1 = input("Give path to template image ")
image_filename2 = input("Give path to test image ")
model = YOLO('best.pt')

results1 = model.predict(image_filename1, save=False, imgsz=320, conf=0.5, hide_labels=False)
results2 = model.predict(image_filename2, save=False, imgsz=320, conf=0.5, hide_labels=False)
if len(results1) > 0 and len(results1[0])>0:
    result1 = results1[0]
    box1 = result1.boxes[0]
    box = box1.xyxy[0]
    x_center1 = float((box[0] + box[2]) / 2)
    y_center1 = float((box[1] + box[3]) / 2)
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])

    print(x_center1,y_center1)

if len(results2) > 0 and len(results2[0])>0:
    result2 = results2[0]
    box1 = result2.boxes[0]
    box = box1.xyxy[0]
    x_center2 = float((box[0] + box[2]) / 2)
    y_center2 = float((box[1] + box[3]) / 2)
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])

    print(x_center2,y_center2)


#Step 2: Cropping
from PIL import Image

image = Image.open(image_filename1)

center_x = x_center1
center_y = y_center1

crop_width = 900
crop_height = 900

left = center_x - (crop_width // 2)
top = center_y - (crop_height // 2)

right = center_x + (crop_width // 2)
bottom = center_y + (crop_height // 2)

cropped_image = image.crop((left, top, right, bottom))

# Save the cropped template image
cropped_image.save("cropped_templateimage.jpg")

from PIL import Image

image = Image.open(image_filename2)

center_x = x_center2
center_y = y_center2

crop_width = 900
crop_height = 900

left = center_x - (crop_width // 2)
top = center_y - (crop_height // 2)

right = center_x + (crop_width // 2)
bottom = center_y + (crop_height // 2)

cropped_image = image.crop((left, top, right, bottom))

# Save the cropped test image
cropped_image.save("cropped_testimage.jpg")

#After this you can use 3 options to detect angular displacement
# 1. Circle.py      2. FLANN.py      3. tempmatch.py
# Circle.py inspired from canny's method works quite good when object is facing upward (convention for direction of object is explained in the pdf)
# FLANN works quite good
# Tempmatch is quite poor here in accuracy

