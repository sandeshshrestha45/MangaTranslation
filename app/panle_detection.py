import cv2
import math
import numpy as np


def display(name, image):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)

# Function to perform non-maximum suppression
def non_max_suppression_fast(boxes, overlapThresh):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list, add the index value to the list of
        # picked indexes, then initialize the suppression list (i.e., indexes that will be deleted)
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # Loop over all indexes in the indexes list
        for pos in range(0, last):
            # Grab the current index
            j = idxs[pos]

            # Find the largest (x, y) coordinates for the start of the bounding box
            # and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # Compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap between the computed bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # If there is sufficient overlap, suppress the current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # Delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)

    # Return only the bounding boxes that were picked
    return boxes[pick]
# Read the image

def main(img_path, root, i):

    image = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

  
    _,thresh = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(image.shape[2],20))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))

    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    threshold_area = sum(cv2.contourArea(area) for area  in outer_contours[:5])/len(outer_contours)
    display("thresh", closing)
    rects = []
    for cnt in outer_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) >= threshold_area:
            # cv2.imshow("crop",image[y:h+y,x:x+w])
            # cv2.waitKey(0)
            rects.append((x,y,w+x,h+y))
    boxes = np.array(rects)
    nms_boxes = non_max_suppression_fast(boxes, overlapThresh=0.6)
    for j,nm_box in enumerate( nms_boxes):
        x,y,w,h = nm_box
        crop_img = image[y:h,x:w]
        # cv2.imwrite(f"{desination_folder}/{i}_{j}_{root}",crop_img)
        cv2.rectangle(image,(x,y),(w,h),(0,0,255),2)
    display("image", image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import glob 
    import os 
    desination_folder = "/home/mrcloud/Documents/projects/caption_data_ne"
    # for root, folder, images in os.walk("/media/mrcloud/New Volume/chulo/comitee-ml/img/one-piece-chapter-1109/"):
    #     for i , image_path in enumerate( images):
    #         file_name = root.split("/")[-1]
    #         img_path  = os.path.join(root, image_path)
    #         print(img_path)
    #     # print(os.path.join(root,images))
    #         file_name += "_"+ image_path
    #         main(img_path, file_name,i)

    img_path = "/home/mrcloud/Documents/projects/manga_download/mangas/new/datasets/Hunter x Hunter v32-ref-HUNTERu╠ê~HUNTERu╠ê@32e╠Ç┬¼197.jpg"
    main(img_path,"s",1)