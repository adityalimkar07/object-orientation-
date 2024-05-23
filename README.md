1. I have sent you best.pt (obtained after training in yolov8) through email as upload limit per file on github is maximum 25 mb. Paste this file in the directory containing python file yolo.py
2. Open either "yolo.py" or "torchv.py" files. Depending on which one you choose, the results may slightly vary or in some cases, an object might not get detected in one or both of the files.
3. Provide the path of the template image and test image as input to the choosen python file.
4. After running the file, it will generate two cropped images: "cropped_templateimage.jpg" and "cropped_testimage.jpg".
5. Next, run either "circle.py", "FLANN.py", or "tempmatch.py".
6. The output of this step will be the rotation of the object in degrees.
