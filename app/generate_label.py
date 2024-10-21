import os
import cv2
from tqdm import tqdm
import shutil
# import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from ultralytics import YOLO


dataset_path = "/home/mrcloud/Documents/projects/manga_download/mangas/new/datasets"
label_path = "/home/mrcloud/Documents/projects/manga_download/mangas/new/label"

model = YOLO("model/chracter/best.pt")

for root, files, images in os.walk(dataset_path):
    for image in tqdm(images):
        label_name = image.split(".jpg")[0]
        label = os.path.join(label_path, label_name+".txt")
        image_path = os.path.join(dataset_path, image)

        detections  = model(image_path,verbose=False)
        # shutil.copy(image_path, os.path.join('/home/mrcloud/Documents/projects/manga_download/mangas/test',image))
        # breakpoint()
        boxes = detections[0].boxes.xywhn
        classes = detections[0].boxes.cls
        with open(label,"a+") as writer:
            for box, cls in zip(boxes, classes):
                if cls ==0 or cls == 1 or cls == 2:
                    x,y,w,h =  box
                    writer.writelines(f"{int(cls)} {x} {y} {w} {h} \n")


