from torchvision.io.image import read_image
import cv2
import numpy as np
import os
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

import matplotlib.pyplot as plt
import requests

image_filename1 = input("Give path to template image ")
img1 = read_image(image_filename1)

weights = FCOS_ResNet50_FPN_Weights.DEFAULT
model = fcos_resnet50_fpn(weights=weights, score_thresh=0.35)
model.eval()

preprocess = weights.transforms()
batch = [preprocess(img1)]
prediction = model(batch)[0]

import numpy as np

boxes = prediction["boxes"].detach().cpu().numpy()

com_coordinates = []
for box in boxes:
    area = (box[0] - box[2])*(box[1] - box[3])
    x_center1 = (box[0] + box[2]) / 2
    y_center1 = (box[1] + box[3]) / 2
    com_coordinates.append((area, x_center1, y_center1))

for i in range(0,len(com_coordinates)):
    com_first_box1 = com_coordinates[i][1:] #template image

image_filename2 = input("Give path to test image ")
img2 = read_image(image_filename2)
weights = FCOS_ResNet50_FPN_Weights.DEFAULT
model = fcos_resnet50_fpn(weights=weights, score_thresh=0.35)
model.eval()

preprocess = weights.transforms()
batch = [preprocess(img2)]
prediction = model(batch)[0]

import numpy as np

boxes = prediction["boxes"].detach().cpu().numpy()

com_coordinates = []
for box in boxes:
    area = (box[0] - box[2])*(box[1] - box[3])
    x_center2 = (box[0] + box[2]) / 2
    y_center2 = (box[1] + box[3]) / 2
    com_coordinates.append((area, x_center2, y_center2))

for i in range(0,len(com_coordinates)):
    com_first_box2 = com_coordinates[i][1:] # test image


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