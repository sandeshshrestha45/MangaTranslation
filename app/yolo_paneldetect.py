from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import config as cfg
from ultralytics.utils.ops import non_max_suppression
import utils as ut
from manga_page_analyzer import MangaPageAnalyzer 
# from gender_detect import get_gender

# ultralytics==8.1.22


# model = YOLO(fg.MODEL_PATH)
# huggin_face_model = YOLO("model/huggin/best.pt")
model = YOLO(cfg.MODEL_PATH)
# model = YOLO("/home/mrcloud/train3/weights/best.pt")
# yolo_added = YOLO("/home/mrcloud/Downloads/arranged_8m-20240524T090744Z-001/yolonew_8m/weights/best.pt")
img_path = "img/Boku no Hero Academia/406/"
models= [model]
# models = [huggin_face_model,model,text_bubble]

def custom_nms(predictions, iou_threshold=0.5):
    if len(predictions) == 0:
        return []
    
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5]
    
    indices = np.argsort(scores)[::-1]
    keep_boxes = []
    
    while len(indices) > 0:
        current = indices[0]
        keep_boxes.append(current)
        
        if len(indices) == 1:
            break
        
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = np.array([iou(current_box, other_box) for other_box in other_boxes])
        
        indices = indices[1:][ious <= iou_threshold]
    
    return predictions[keep_boxes]


def iou(box1, box2):
    x1, y1 = np.maximum(box1[:2], box2[:2])
    x2, y2 = np.minimum(box1[2:4], box2[2:4])
    
    inter_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    iou = inter_area / (box1_area + box2_area - inter_area)
    
    return iou

def draw_mask (masks):
    
    for mask in masks:
        # print(mask[0])
        mask_coordinates = np.array(mask, dtype=np.int32)
        mask_coordinates = mask_coordinates.reshape((-1, 1, 2))
        # mask = [int(cordinate ) for cordinate in mask]
        # x,y = mask
        cv2.fillPoly(img_mask,[mask_coordinates],(255,255,255),1)


def parse_rect(box, x1,y1,color):
    if box:
        tx = box[0]+x1
        ty = box[1]+y1
        tw =  box[2]+x1
        th = box[3]+y1
    # print(text_box)
        # cv2.rectangle(img,(tx,ty),(tw,th),color,2)
    return tx, ty, tw, th

def adjust_text_box(bounding_box, segmented_text_box):
    x1, y1, x2, y2 = bounding_box
    sx1, sy1, sx2, sy2 = segmented_text_box

    # Calculate the height of the bounding box
    bounding_box_height = y2 - y1

    # Calculate 20% of the bounding box's height
    threshold = 0.2 * bounding_box_height

    # Check if the start point of the segmented box is greater than 20% of the bounding box size
    if sy1 - y1 > threshold:
        # Decrease the start point of the segmented text box's Y-coordinate
        print("distance  ", sx1,y1, bounding_box_height, threshold)
        sy1 = y1 + int(threshold)

    # Return the adjusted segmented text box
    return sx1, sy1, sx2, sy2

def draw_box(box,color_code, class_name):
    text_boxes = []
    bounding_box = []
    k= 0
    for i, box in enumerate( box):
        # box = [int(b) for b in box]
        x1, y1, x2, y2, acc, index = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        index = int(index)
        # print(acc)
        if acc >= 0.5 :
            crop_img = img[y1:y2,x1:x2]
          
            if index == 0 or index == 1:
                k += 1
                # print("index ",k,acc, index)
                # cv2.imwrite(f"img/translated_img/tokyo/black{i}.jpg",crop_img)
                try:
                    text_box = ut.segment_text_from_bubble(crop_img.copy())
                    # polygon_rect = get_polygon_inside_rect(crop_img.copy())
                    bounding_box.append([ x1, y1, x2, y2])
                    text_bbox =parse_rect(text_box,x1,y1,(0,255,0))
                    x,y,w,h = adjust_text_box(bounding_box=[x1, y1, x2, y2],
                                              segmented_text_box= text_bbox
                                              )
                    text_boxes.append([x,y,w,h])
                    # # parse_rect(polygon_rect,x1,y1,(0,0,255))
                    print(index,"----", type(index))
                    cv2.putText(img, str(index) + "__" +str(acc),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2 )
                    # cv2.rectangle(img,(x1,y1),(x2,y2),color_code,1)
                    # cv2.imshow("crop",crop_img)
                    # print(crop_img.shape)
                    # cv2.waitKey(0)
                    cv2.imwrite(f"img/translated_img/large{k}.jpg",crop_img)
                except Exception as e:
                    # raise
                    cv2.imshow("crop",crop_img)
                    cv2.imwrite("img/translated_img/tokyo/large.jpg",crop_img)
                    # raise
                    # breakpoint()
                    print(e)

            else:
                pass
                # cv2.rectangle(img,(x1,y1),(x2,y2),color_code,1)

            # cv2.putText(img,  class_name[index]+ f"_{str(i)}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,color_code,2)
    for box, bounding_box in zip(text_boxes, bounding_box):
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),1)
        cv2.rectangle(img,(bounding_box[0],bounding_box[1]),(bounding_box[2],bounding_box[3]),(0,255,0),1)
        new_x = box[0] - (box[0] - bounding_box[0]) //2
        new_x2 = box[2] +(bounding_box[2]  -box[2] )//2
        
        # print(f"x {box[0]} bx {bounding_box[0]} new_x {new_x}  --   x2 {box[2]} bx2 {bounding_box[2]} new_2 {new_x}]")
        cv2.rectangle(img,(new_x,box[1]),(new_x2,box[3]),(0,0,255),1)

def sort_panels_by_reading_order(panels):
    # Sort by Y coordinate, then by X coordinate (assuming right-to-left, top-to-bottom reading order)
    return sorted(panels, key=lambda p: (p.y, -p.x))




for images in os.listdir(img_path):
    print(images)
    if ".jpg" not in images:
        continue

    img = cv2.imread(os.path.join(img_path,images))
    img = cv2.imread("/home/mrcloud/Downloads/[赤坂アカ] かぐや様は告らせたい～天才たちの恋愛頭脳戦～ 第01巻/0003.jpg")
    # img = cv2.blur(img,(3,3))
    img_mask= np.zeros(img.shape[:2], dtype=np.uint8)
    for i, model in enumerate( models):

        detections  = model.predict(img,conf=0.5)
        # breakpoint()
        predictions  = detections[0].boxes.data.cpu().numpy()  # Assuming batch size of 1
        print(predictions, len(predictions))
        filtered_predictions = custom_nms(predictions, iou_threshold=0.4)
        # print(filtered_predictions,len(filtered_predictions))

        # print(filtered_predictions)
        # text_bubble_detection = text_bubble.predict(img,iou=0.4)
        # text_bubble_box = detections[0].boxes.data.cpu().numpy()
        # breakpoint()

        class_names = detections[0].names

        # class_names_text_bubble = text_bubble_detection[0].names
        box  = detections[0].boxes.data.cpu().numpy()
        # detections= non_max_suppression(box, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300, nc=0, max_time_img=0.05, max_nms=30000, max_wh=7680, in_place=True, rotated=False)
        '''
        if i == 0:
            if detections[0]:
                mask = detections[0].masks.xy
                draw_mask(mask)
        '''
        # text_buuble_box = text_bubble_detection[0].boxes.data.cpu().numpy()
        # print(class_names_text_bubble)
        

        # color_code = np.random.uniform(0, 255, size=(len(class_names), 3))
        # color_code = (255,0,0)
        # text_buuble_color = np.random.uniform(0, 255, size=(len(class_names_text_bubble), 3))
        # text_bubble_color = (0,255,0)
        panels =[]
        # [panels.append(panel) for panel in box if box[-1]==2]
        # breakpoint()
        if i == 0:
            color_code = (255,0,0)
        elif i == 1:
            color_code = (0,255,0)
        else:
            color_code = (0,0,255) 

        draw_box(filtered_predictions,color_code, class_name=class_names)
        # draw_box(text_buuble_box, text_bubble_color, class_name= class_names_text_bubble)
    # result = cv2.addWeighted(img, 0.7, img_mask, 0.3, 0)
    # img = cv2.inpaint(img,img_mask,7,cv2.INPAINT_TELEA)
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imwrite("img/translated_img/blur.jpg",img)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break