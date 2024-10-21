import cv2
import re
import os
import math
import numpy as np
import ast
from PIL import Image
from collections import Counter
import shutil
import json
from google.cloud import storage
from PIL import Image
import config as cfg
from config import logger_ocr
import traceback

def delete_folder(path):
    """
    Delete all the content of the folder path
    """
    try:
        shutil.rmtree(path)
        print(f"All contents along with the directory {path} have been deleted")
    except Exception as e:
        save_error(f"Error ar delete_folder {e}")
        print(f"Failed to delete {path}: {e}")
        
def sort_panels_by_reading_order(panels):
    """
    Sorts manga panels by reading order based on their positions on the page.

    This function sorts a list of manga panels according to their reading order, which typically follows a top-to-bottom and right-to-left sequence. The sorting is done based on the y-coordinate (top-to-bottom) and x-coordinate (right-to-left) of the panels.

    Parameters:
        - panels (list): A list of `DetectedObject` instances representing the manga panels to be sorted.

    Returns:
        - list: The sorted list of `DetectedObject` instances in the correct reading order.

    Sorting Logic:
        - The primary sorting is by the y-coordinate (top-to-bottom).
        - If two panels have the same y-coordinate, they are further sorted by the x-coordinate in descending order (right-to-left).
    """

    return sorted(panels, key=lambda p: (p.y, -p.x))

def sort_text_bubble_by_reading_order(panels):
    return sorted(panels, key=lambda p: -p.x)

def intersects(obj1, obj2):
    """
    This function checks if the detected text bubble or characters lies on which panel
    """
    if (obj1.x > obj2.x and 
        obj1.width < obj2.width and
        obj1.y > obj2.y and 
        obj1.height < obj2.height):
        return True
    return False

# Function to assign each bubble and character to a panel
def assign_to_panel(bubbles_chars, panels):
    """
    This funtion assign chracters and text bubbles to the panel by checking intesection points.

    """
    for item in bubbles_chars:
        for i, panel in enumerate(panels):
            if intersects(item, panel):
                item.panel_index = i


def handel_panel_less(panel_less,panels):
    """
    It panel id to 0 if their is no panel on the manga page
    """
    for no_panel in panel_less:
        distance= []
        if len(panels) != 0:
            for i,panel in  enumerate(panels):
                distance.append([math.hypot(panel.x - no_panel.x, panel.y - no_panel.y),i])
        
            sorted_points = sorted(distance, key=lambda x: x[0])
            # print(sorted_points, no_panel.x)

            no_panel.panel_index = sorted_points[0][1]
        else:
            no_panel.panel_index = 0

def convert_cv2_image_to_pil(cv2_image):
    # Convert the color space from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert the OpenCV image (NumPy array) to a PIL image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

def parase_text_box(box, x1,y1):
    """
    Adjusts the coordinates of a text box relative to a given offset.
    This function takes a bounding box (representing a text box) and adjusts its coordinates based on the provided offsets `(x1, y1)`. 
    The adjusted coordinates are returned as a new bounding box.
    """
    if box:
        tx = box[0]+x1
        ty = box[1]+y1
        tw =  box[2]+x1
        th = box[3]+y1
        return tx, ty, tw, th
    
def get_text(objects,image,id, text_id,mocr):
    """
    Extract text and segement the text cordinates from the detected text bubble.
    It segment the text from the text bubble only if it the text bubble not the text free (text without bubble)
    Parameters:
        - objects (list): A list of `DetectedObject` instances representing the detected text bubbles or characters within the manga image.
        - image (numpy.ndarray): The original manga image from which text is to be extracted.
        - id (str): A unique identifier for the image or panel being processed.
        - text_id (dict): A dictionary to store the mapping between unique IDs and the corresponding detected objects with their extracted text.
        - mocr (function): The OCR model or function used to extract text from the cropped image.

    Returns:
        - original_texts (dict): A dictionary containing the extracted text for each object, where keys are unique IDs and values are the extracted text.
    """
    original_texts = {}
    for i,obj in enumerate(objects):
        crop_img = image[int(obj.y):int(obj.height), int(obj.x): int(obj.width)]
        if obj.cls_index !=1:
            text_box = parase_text_box(segment_text_from_bubble(crop_img.copy()), obj.x, obj.y)
            obj.text_cooordinate = text_box
            # crop_img = image[text_box[1]:text_box[3], text_box[0]:text_box[2]]
        else:
            obj.text_cooordinate = [obj.x, obj.y, obj.width, obj.height]
        original_text = mocr(convert_cv2_image_to_pil(crop_img))
        # print("Mocr ", original_text )
        # logger_ocr.info(original_text) # printing log
        uq_id = f"{id}_{i}" if obj.cls_index==0 else f"{id}_{i}_f"
        text_id.update({uq_id:[obj, original_text+ " "]})
        original_texts.update({uq_id: original_text+ " "})
    return original_texts

def parase_character( characters,clases_index):
    """
    Summarizes the detected characters by gender within a manga panel.

    This function processes a list of detected characters, counts the number of characters for each gender, 
    and generates a summary string describing the count of each gender within the panel.
    Returns:
    - char_string (str): A string summarizing the count of characters by gender, e.g., "2 number of boy 3 number of girl".

    """
    chracter=  [mg_character.gender for mg_character in characters ]
    char_string = ""
    chracter = Counter(chracter)
    for k , v in chracter.items():
        char_string += f"{v} number of {k} "
    return char_string

def calculate_local_contrast(image, block_size):
    """ Calculate local standard deviation across the image. """
    local_std = np.zeros(image.shape)
    for y in range(0, image.shape[0], block_size):
        for x in range(0, image.shape[1], block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                local_std[y:y+block_size, x:x+block_size] = np.std(block)
    return local_std

def decide_thresholding(image, block_size=32):
    """ Decide whether to use global or adaptive thresholding based on local contrast. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local_std = calculate_local_contrast(gray, block_size)
    global_std = np.std(gray)

    # Simple heuristic: if the max local std deviation is significantly larger than the global std, use adaptive
    if np.max(local_std) > 1.5 * global_std:
        return 'adaptive'
    else:
        return 'global'

def calculate_threshold_based_on_brightness_contrast(image):
    """Determine threshold based on the brightness and contrast of the image."""
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()

    # Define thresholds based on empirical observation or testing
    if brightness > 180 and contrast < 75:  # Adjust these values as needed
        return 220  # Higher threshold for bright and low-contrast images
    elif contrast >84:

        return 146  # Lower threshold for high contrast images
 # Lower threshold for high contrast images
    else:
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ret if ret > 127 else 180 


def detect_shape(image):
    """
    Detect shapes of the bubble 
    """
   
    image_mask = image > 127
    contours,_= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea
    )

    contour_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Fit ellipse if there are enough points and check the fit
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            ellipse_mask = np.zeros_like(image)
            cv2.ellipse(ellipse_mask, ellipse, (255, 255, 255), -1)
            intersection = np.logical_and(ellipse_mask, image_mask)
            union = np.logical_or(ellipse_mask, image_mask)
            iou = np.sum(intersection) / np.sum(union)
            if iou > 0.85:  # Adjust IoU threshold for better fitting
                return "Oval"
        except cv2.error:
            pass

    # Fit a rotated rectangle and check for match
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    rect_mask = np.zeros_like(image)
    cv2.drawContours(rect_mask, [box], 0, (255, 255, 255), -1)
    intersection = np.logical_and(rect_mask, image_mask)
    union = np.logical_or(rect_mask, image_mask)
    iou = np.sum(intersection) / np.sum(union)
    if iou > 0.85:  # Adjust IoU threshold for better fitting
        return "Rectangle"

    # Check for irregularity by analyzing the contour's complexity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area

    if solidity < 0.8:  # Solidity threshold can be adjusted
        return "Irregular"

    return "Polygon"

def fill_text_bubble (bubble, key):
    """
    Fills the text area in a manga bubble with the background color for text replacement.

    This function processes a detected text bubble in a manga image, determining the appropriate thresholding method to isolate the text. It then fills the text area with a background color, preparing the bubble for overlaying translated text. The function also detects the shape of the text bubble for further processing.
    Workflow:
        1. Determines the appropriate thresholding method for the `bubble` region based on its brightness and contrast.
        2. Converts the `bubble` to grayscale and calculates its average intensity to decide on background and text colors.
        3. Applies either adaptive or standard thresholding to isolate the text region.
        4. Finds the contours of the text regions and selects the largest contour as the primary text area.
        5. Dynamically filters out smaller contours to focus on significant text areas, filling these with the background color.
        6. If the `key` indicates a text free  the fill color is dynamically determined based on the median color in the region.
        7. Detects the shape of the text bubble using the processed mask.
        8. Returns the filled image region, the determined text color, and the shape of the text bubble.
    """

    thresholding_method = decide_thresholding(bubble)
    mask = np.zeros_like(bubble)
    # Convert to grayscale
    gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(f"img/translated_img/tokyo/{key}.jpg",gray)
    # gray = cv2.blur(gray,(5,5))
    # cv2.imshow("gray",gray)
    box_h,box_w = gray.shape
    avg_intensity = np.mean(gray)
    intensity_threshold = 0
    method = ""
    if avg_intensity > 117:
        background_fill = (255,255,255)
        text_color = (0,0,0)
        constant = int(0.02*box_w)
        method = cv2.THRESH_BINARY
        intensity_threshold = 100
        
        # _, thresh = cv2.threshold(gray, 222, 255, cv2.THRESH_BINARY)
    else:
        background_fill = (0,0,0)
        text_color = (255,255,255)
        intensity_threshold = 50
        # constant = int(0.09*box_h)
        constant = int(0.01*box_w)
        method = cv2.THRESH_BINARY_INV
        # _, thresh = cv2.threshold(gray, 222, 255, cv2.THRESH_BINARY_INV)

    # Use adaptive thresholding to accommodate different lighting conditions
    if thresholding_method == "adaptive":
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            method, 21, constant)
    else:
        thresh_value= calculate_threshold_based_on_brightness_contrast(bubble)
        ret,thresh = cv2.threshold(gray,thresh_value,255, method)
        # print(thresh_value)
        # cv2.imshow("thresh",thresh)
        # cv2.waitKey(0)

    # if thresholding_method == "adaptive":
    #     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                     cv2.THRESH_BINARY, 21, constant)
    # else:
    #     _,thresh = cv2.threshold(gray,220,255, method)
    # if thresholding_method =="adaptive":
    #     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                 cv2.THRESH_BINARY, 21, constant)
    # else:
    #    if avg_intensity <127:
            
    #         _,thresh = cv2.threshold(gray,220,255, cv2.THRESH_BINARY_INV)
    #    else:
    #        _,thresh = cv2.threshold(gray,220,255, cv2.THRESH_BINARY)
           
   
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume text is the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    result = bubble.copy() 
    # cv2.fillPoly(result, [largest_contour], background_fill) 
    # return result, text_color


    # Fill the largest contour with a color that matches the background
    # cv2.fillPoly(result,largest_contour, background_fill)
    """
        The following code is to fill the japanese text which has two or more contour 
        i.e the japanese text seprated by the distanse
    """
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    avg_contour_area = np.mean(contour_areas)
    std_contour_area = np.std(contour_areas)
    dynamic_area_threshold = avg_contour_area + 0.5 * std_contour_area
    for cnt in contours: 
        if  cv2.contourArea(cnt) >= dynamic_area_threshold:
            if "f" in key:
                # mask = np.zeros(bubble.shape[:2], dtype=np.uint8)
                # cv2.fillPoly(mask, [largest_contour], color=background_fill)
                roi_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                roi_flat = roi_rgb.reshape(-1, 3)

                # # Find the most common color in the ROI
                fill_color = np.median(roi_flat, axis=0).astype(np.uint8)
                text_color = (0,0,0)

                
                cv2.fillPoly(result,[cnt], fill_color.tolist())
                cv2.fillPoly(mask,[cnt],(255,255,255))
            else:
                cv2.fillPoly(result, [cnt], background_fill)  # You may want to replace this with a dynamic background color detection
                cv2.fillPoly(mask,[cnt],(255,255,255))
    
    bubble_shape = detect_shape(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY))
    hist = cv2.calcHist([result], [0], None, [256], [0, 256])
    max_intensity = np.argmax(hist)
    if max_intensity< intensity_threshold and "f" in key:
        cv2.rectangle(result, (0,0),(result.shape[1],result.shape[0]), fill_color.tolist(),-1)
    # cv2.imshow("result",result)
    
    # cv2.imshow("thresh",thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result, text_color, bubble_shape
"""
def merge_contour_rectangles(contours, w,h):
    # Calculate image center
    img_center_x, img_center_y =w// 2, h // 2

    # Store the bounding rectangles and centers
    rectangles = [cv2.boundingRect(cnt) for cnt in contours]
    centers = [(x + w // 2, y + h // 2) for x, y, w, h in rectangles]

    # Determine the threshold for "centeredness", here using 10% of image width/height
    threshold_x =w* 0.25
    threshold_y = h * 0.27

    # Collect centered rectangles
    centered_rects = [rect for rect, center in zip(rectangles, centers)
                      if abs(center[0] - img_center_x) <= threshold_x and
                         abs(center[1] - img_center_y) <= threshold_y]

    # Merge rectangles that are considered centered
    if not centered_rects:
        return None  # No centered rectangles found
    
    # Calculate the bounding box of the centered rectangles
    x_min = min(rect[0] for rect in centered_rects)
    y_min = min(rect[1] for rect in centered_rects)
    x_max = max(rect[0] + rect[2] for rect in centered_rects)
    y_max = max(rect[1] + rect[3] for rect in centered_rects)

    # Return the combined rectangle
    return (x_min, y_min, x_max , y_max)
"""

def calculate_new_bounding_box(original_bbox, segmented_bbox, distance_ratio=0.2):
    """
    Calculates a new bounding box by adjusting the original bounding box based on the segmented bounding box and a distance ratio.

    This function adjusts the coordinates and dimensions of an original bounding box based on the positions of a smaller, segmented bounding box inside it. 
    The adjustment is scaled by a specified distance ratio, resulting in a new bounding box that better aligns with the segmented region.

    Parameters:
        - original_bbox (list or tuple): The coordinates and dimensions of the original bounding box in the format `[x, y, width, height]`.
        - segmented_bbox (list or tuple): The coordinates and dimensions of the segmented bounding box inside the original bounding box in the format `[x, y, width, height]`.
        - distance_ratio (float, optional): The ratio used to adjust the distances between the original and segmented bounding boxes. Default is `0.2` (20%).

    Returns:
        - list: A list containing the new bounding box coordinates and dimensions in the format `[new_x, new_y, new_width, new_height]`.

    Workflow:
        1. Extracts the distances between the edges of the original bounding box and the segmented bounding box on all four sides (left, top, right, bottom).
        2. Calculates the adjusted distances based on the specified `distance_ratio`.
        3. Computes the new coordinates and dimensions of the bounding box by applying the adjusted distances.
        4. Returns the new bounding box as a list of integers.
    """
    # Extract coordinates and dimensions of the bounding boxes
    left_distance = segmented_bbox[0] - original_bbox[0]
    top_distance = segmented_bbox[1] - original_bbox[1]
    right_distance = (original_bbox[0] + original_bbox[2]) - (segmented_bbox[0] + segmented_bbox[2])
    bottom_distance = (original_bbox[1] + original_bbox[3]) - (segmented_bbox[1] + segmented_bbox[3])

    # Calculate the new distances (20% of the distance between bigger and smaller rectangles)
    new_left_distance = left_distance * distance_ratio
    new_top_distance = top_distance * distance_ratio
    new_right_distance = right_distance * distance_ratio
    new_bottom_distance = bottom_distance * distance_ratio

    # Calculate the coordinates of the new rectangle
    new_x = original_bbox[0] + new_left_distance
    new_y = original_bbox[1] + new_top_distance
    new_width = original_bbox[2] - (new_left_distance + new_right_distance)
    new_height = original_bbox[3] - (new_top_distance + new_bottom_distance)

    return [int(cordinates) for cordinates in (new_x,new_y,new_width,new_height)]

def segment_text_from_bubble(bubble):
    """
    Segments the text region from a manga text bubble by isolating the text area using morphological operations.

    This function processes an image of a manga text bubble to isolate and segment the text area within it. It applies image subtraction and morphological operations to identify the contours of the text and returns the bounding box that tightly encloses the text region.

    Parameters:
        - bubble (numpy.ndarray): The image region containing the manga text bubble.

    Returns:
        - list: A list containing the coordinates and dimensions `[x_min, y_min, x_max, y_max]` of the bounding box that encloses the segmented text region.
        - If no contours are found, the function returns the original bounding box `[0, 0, w, h]`.

    Workflow:
        1. Creates a copy of the input `bubble` image.
        2. Applies the `fill_text_bubble` function to the `bubble` image, which fills the text region with a background color.
        3. Subtracts the filled image from the original image to highlight the text region.
        4. Converts the result to grayscale and applies thresholding to create a binary image.
        5. Uses morphological operations to clean up the binary image and isolate the text contours.
        6. Iterates over the detected contours to determine the bounding box that tightly encloses all the text.
        7. Returns the bounding box coordinates. If no contours are found, it returns the entire original bounding box.
    """
    bubble_copy=  bubble.copy()
    h,w,_ = bubble.shape
    bubble,_,_ = fill_text_bubble (bubble,key="")
    sub_img = cv2.bitwise_xor( bubble_copy,bubble)
    sub_img = cv2.cvtColor(sub_img,cv2.COLOR_BGR2GRAY)
    _,thresh =cv2.threshold(sub_img,220,255,cv2.THRESH_BINARY)

    # kernel =  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))


    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3,3))
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min = float('inf')
    y_min = float('inf')
    x_max = 0
    y_max = 0
    if len(contours) >0:
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + cw)
            y_max = max(y_max, y + ch)
    else:
        return [0, 0, w, h]

    return [x_min, y_min, x_max, y_max]
       

def draw_on_image(img,detections,clases_index,name):
    temp =  img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale =0.5
    font_thickness = 1
    text_color = (0, 0, 255) 
    z= 0
    for i ,bounding_box in enumerate(detections):
        
        x,y,w,h, cls_index, panel_id = int(bounding_box.x), int(bounding_box.y),int(bounding_box.width),int(bounding_box.height), int(bounding_box.cls_index), str(bounding_box.panel_index)
        if clases_index[cls_index] == "panel":
            panel_id = str(i)
        # cv2.putText(temp,clases_index[cls_index]+ "_"+ panel_id+ "-"+str(i), (x+10,y+20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        # cv2.rectangle(temp,(x,y),(w,h),(0,0,255), 1)
        crop = temp[y:h,x:w]
        cv2.imwrite(f"img/translated_img/{name}_{z}_bokuno.png", crop)
        z += 1
    cv2.imwrite(f"img/translated_img/{name}_bokuno.png",temp)

def extract_japanese_text(data):    
    """
    Recursively extracts Japanese text from a nested dictionary structure.

    This function traverses a dictionary that contains data about manga panels and extracts all Japanese text found within. The extracted texts are returned as a new dictionary, where each key corresponds to a specific text segment.

    Parameters:
        - data (dict): A nested dictionary structure containing manga panel data, which may include Japanese text under various keys.

    Workflow:
        1. Initializes an empty dictionary `japanese_texts` to store the extracted text.
        2. Iterates over the key-value pairs in the input `data`.
        3. If the value is not "this panel doesn’t contain text" and the key is `japanese_text`, it checks the type of the value:
            - If the value is a dictionary, it merges its content into `japanese_texts`.
            - If the value is a string, it assigns the text to a unique key in `japanese_texts`.
        4. Recursively processes any nested dictionaries to extract more Japanese text.
        5. Handles special cases such as lists of text or panels without text.
        6. Returns the final `japanese_texts` dictionary containing all extracted Japanese text.

    Returns:
        - japanese_texts (dict): A dictionary containing all the extracted Japanese text, with each piece of text assigned a unique key.
    """

    japanese_texts = {}
    for key, value in data.items():
        if value != "this panel doesn’t contain text":
            if key == 'japanese_text' :
                if isinstance(value, dict):
                    japanese_texts.update(value)
                elif isinstance(value, str):
                    # Assuming a placeholder here as the key should be unique; using the path as a key
                    print("insise this --- ", value)
                    japanese_texts[f'text_{len(japanese_texts)}'] = value
            elif isinstance(value, dict):
                japanese_texts.update(extract_japanese_text(value))
            elif key == "only_text" :
                if isinstance(value,list):
                    for data in value:
                        japanese_texts.update(data)
            elif key == "panel_less" and len(value) > 0:
                japanese_texts.update(value[0]["japanese_text"])
       
    return japanese_texts

def parser_and_validate_story_prompt(response,client,max_retrires=3):
    """
    It validate the story response return from the gpt api is correct or not. If not it retries 3 times till it get correct response else return empty sotry
    """
    retries = 0 
    while retries < max_retrires:
        try:

            story = response["story"]

            # print("inside Story parse prompt")
            return response
        except Exception as e:
            story_format ={"story":"values"}
            save_error(f"Error at parser_and_validate_story_prompt {e}")
            retries +=1
            if retries <max_retrires:
                logger_ocr.error(f" Story generation error number of retries {retries}. The error is {e} ")
                correction_prompt = f" There is and error {e} in this data {response} convert this it into this format {story_format} :Note return only corrected json format"
                response = translate_chatgpt(system_prompt=response,user_prompt=correction_prompt, client=client)
            else:
                return {"story":{}}
    

def parse_and_validate_context(response, target_language, prompt_keys,client,max_retrires=3):
    """
    It validate the translation response return from the gpt api is correct or not. If not it retries 3 times till it get correct response else return none
    """
    format= {}
    for tlanguage in target_language:
        keys_data= {}
        for pkey in prompt_keys:
            keys_data.update({pkey:"translated text"})
        format.update({tlanguage:keys_data})
    retries = 0 
    while retries < max_retrires:
        try:
            for  tlanguage in target_language:
                # data = response[tlanguage]
                for pkeys in prompt_keys:
                   continue
                    # print(response[tlanguage][pkeys], pkeys)
            return response
        except Exception as e:
            # raise
            save_error(f"Error at parse_and_validate_context {e}")
            logger_ocr.debug(f" Error in parase context the required Format is: {format}. got this error : {e} ")
            
            retries += 1
            if retries <max_retrires:
                logger_ocr.error(f" Error in trainslation number of retires {retries}. The error is {e}")
                correction_prompt = f""" {response} This json  output  of and transaltion api. The json output has this {e} error correct this and return on this format {format}.\
                    Note:  
                    - Return only json data and 'translate_key' an placeholder of the translated text  replace with the respective value from the json data.\
                    - If the key are missing extract the respective japanese text from the manga_context of the context.\
                    - Do not extract japanese text from the similar_context."""   
                response = translate_chatgpt(system_prompt= response, user_prompt=correction_prompt, client=client)
                # print("Corrected response", response)
            else:
                return {}

def translate_chatgpt(system_prompt,user_prompt, client,max_retrires= 3):
    """
        Function to interact with ChatGPT and handle JSON format errors with a retry mechanism.

        Parameters:
        - prompt (str): The initial prompt to send to ChatGPT.
        - client (object): The client object to interact with ChatGPT's API.
        - max_retries (int): The maximum number of retries allowed in case of JSON format errors.

        Returns:
        - data (dict or None): The parsed JSON data returned by ChatGPT, or None if max retries are reached.
    """

    retries = 0 
    response =""
    while retries < max_retrires:
        # print("retires", retries, " *** "*8)
        try:
            response = client.chat.completions.create(
                        model=cfg.GPT_MODEL,
                        messages= [
                            {
                                "role":"system",
                                "content": system_prompt
                            },
                            {
                                "role":"user",
                                "content":user_prompt
                            }
                        ],
                        temperature=0.3,
                        max_tokens=4096,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
            data = response.choices[0].message.content.replace("```json", "").replace("```", "")

            # print(f"Response {type(data)}----------------  \n",data)
            # logger_ocr.info(f" Chat gpt response: \n {data}")
            data = ast.literal_eval(data) if (isinstance(data,str)) else json.dumps(data)
            return data
        except Exception as e:
            
            save_error(f"Error at translate_chatgpt {e}")

            retries +=1 
            # print(f"Retires value error -----------------,  {e}", retries)
            logger_ocr.error(f"Error at translate_chatgpt function  Retires value error -----------------  {e} number of retries {retries}")
            if retries < max_retrires:
                if response:
                    user_prompt = f"There is an error in this JSON format. got this error {e} Correct the error: of the given json :\n {response.choices[0].message.content} return the corrected json format return json format only do not add '```' on th response Note dont add additional keys or data only correct the data response shoudl only contain as given data i,e {data}" 
                else :
                    user_prompt = user_prompt
            else:
                logger_ocr.debug(" Maximum retries reached. Exiting.")
                # print("Maximum retries reached. Exiting.")
                return None
            

def get_weaviet_schema(weaviet_client):
    """
    Retrieves or creates the Weaviate schema for storing manga-related data.

    This function checks if a specific Weaviate schema exists on the given Weaviate client. 
    If the schema does not exist, it creates the schema with the specified properties. 
    The schema is designed to store information about manga, including the manga name, title, chapter, page number, story, and context. 
    The function also configures vectorization using the `text2vec-openai` module with the specified embedding model.
    """
    class_obj = {"class":cfg.WEAVIET_SCHEMA , 
                "properties":[
                    {
                        "name":"manga_name",
                        "dataType": ["text"]
                    },
                    {
                        "name": "title",
                        "dataType": ["text"]
                    },
                    {
                        "name": "chapter",
                        "dataType":["text"]
                    },
                    {
                        "name": "page_no",
                        "dataType": ["text"]
                    },
                    {
                        "name": "story",
                        "dataType":["text"],
                        "vectorize": True
                    },
                    {
                        "name": "context",
                        "dataType": ["text"],
                        "vectorize": True
                    }
                ],
                "vectorizer": "text2vec-openai",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "model": cfg.EBEDDING_MODEL,
                            "options": {
                                "waitForModel": True,
                                "useCache": True
                            }
                        }
                    }
                }
    try: 
        return weaviet_client.schema.get(cfg.WEAVIET_SCHEMA)
    except Exception as e:
        save_error(f"Error at get_weaviet_schema {e}")


        weaviet_client.schema.create_class(class_obj)

        return weaviet_client.schema.get(cfg.WEAVIET_SCHEMA)

def get_record_uuid(client, manga_name, title, chapter, page_no):
    """
    Retrieves the UUID of a specific record from the Weaviate database based on manga details.

    This function queries the Weaviate database for a record that matches the given manga name, title, chapter, and page number. 
    If a matching record is found, the function returns the UUID of that record. If no match is found or an error occurs, the function returns `None`.

    """
    query = """
    {
        Get {
            Manga(where: {
                operator: And,
                operands: [
                    {path: ["manga_name"], operator: Equal, valueText: "$manga_name"},
                    {path: ["title"], operator: Equal, valueText: "$title"},
                    {path: ["chapter"], operator: Equal, valueText: "$chapter"},
                    {path: ["page_no"], operator: Equal, valueText: "$page_no"}
                ]
            }) {
                _additional {id}
            }
        }
    }
    """.replace("$manga_name", manga_name).replace("$title", title).replace("$chapter", chapter).replace("$page_no", page_no)
    try:
        result = client.query.raw(query)
        records = result["data"]["Get"]["Manga"]
        if records:
            return records[0]["_additional"]["id"]
        return None
    except Exception as e:
        print(f" Error at record adding: {e}")
        save_error(f"Error at get_record_uuid {e}")

        return None

def add_record_to_weaviet(weaviet_client, story_data):
    """
    Add record to weaviate database. 
    If uuid found in the database it update the record else it insert new record into the database.
    """
    try:
        uuid = get_record_uuid(weaviet_client, story_data["manga_name"],
                            story_data["title"],
                                story_data["chapter"],
                                story_data["page_no"])
        question_object = {
            "manga_name": story_data.get("manga_name",""),
            "title": story_data.get("title",""),
            "chapter": story_data.get("chapter",""),
            "page_no": story_data.get("page_no",""),
            "story": story_data.get("story",""),
            "context": json.dumps(story_data.get("context",""))
            }
        if uuid is None:
            weaviet_client.data_object.create(question_object, class_name=cfg.WEAVIET_SCHEMA)
            logger_ocr.info(f"\033[32m data added: page no {story_data['page_no']} \033[0m ")
            # print(f"\033[32m data added: page no {story_data['page_no']} \033[0m ")
        else:
            weaviet_client.data_object.update(question_object, uuid=uuid, class_name=cfg.WEAVIET_SCHEMA)
            logger_ocr.info(f"\033[32m data updated: page no {story_data['page_no']} \033[0m ")
    except Exception as e:
        save_error(f"Error at add_record_to_weaviet {e}")

        logger_ocr.error(f"Error at adding record in waviate::  {e}")
        


def save_error(error):
    with open(cfg.ERROR_FILE,"a") as f:
        f.write(f"{str(error)}\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        f.write("\n---\n")  # Separate entries


def get_similar_context(weaviet_client, story_data):
    """
    Retrieves a similar context from the Weaviate database based on the provided story data.

    The function uses semantic search to find records that are close in meaning to the input `story_data["story"]`. 
    It compares to the record having same manga name, chapter and diffrent page number.

    Returns:
        - dict: The context of the most similar record found, or an empty dictionary if no similar context is found or an error occurs.

    """
    retry_count = 0
    max_retries= 2
    while retry_count <=max_retries:

        try :
            response = weaviet_client.query.get(
                cfg.WEAVIET_SCHEMA,
                ["title","chapter","page_no","story","context","_additional { certainty }"])\
                .with_near_text({"concepts":story_data["story"]})\
                .with_limit(2)\
                .with_where({
                    "operator": "And",
                            "operands":[
                                {
                                    "path":["manga_name"],
                                    "operator": "Equal",
                                    "valueText": story_data["manga_name"]
                                },
                                {
                                    "path":["chapter"],
                                    "operator": "Equal",
                                    "valueText": str(story_data["chapter"])
                                },
                                {
                                    "path":["page_no"],
                                    "operator": "NotEqual",
                                    "valueText": str(story_data["page_no"])
                                }
                            ],
                }).do()
            data = json.loads(response["data"]["Get"]["Manga"][0]["context"])
            return data
            # logger_ocr.info(f" Similar context \n  {data}")
        except Exception as e:
            save_error(f"Error at get_similar_context {e}")
            
            retry_count += 1
            # raise
            data ={}
            logger_ocr.info(f"\033[31m  Error in retriving Similar context: retries {retry_count} error: {e} \033[0m")
            if retry_count >= max_retries:
                logger_ocr.info(f"\033[31m Max retries reached. Could not retrieve similar context. \033[0m")
                break
    return data


def format_target_language(target_language):
    """
    Formats a list of target languages into a readable string.

    This function takes a list of target languages and formats it into a single string. If the list contains only one language, it returns that language as a string. 
    If there are multiple languages, it joins them with commas and adds 'and' before the last language to create a natural language list.
    Parameters:
    - target_language (list): A list of strings representing the target languages.

    Returns:
        - str: A formatted string representing the target languages. If the list contains only one language, that language is returned. 
        If there are multiple languages, they are joined with commas and 'and'.
    Examples:
        - `format_target_language(['en'])` returns `'en'`.
        - `format_target_language(['en', 'zh'])` returns `'en and zh'`.
        - `format_target_language(['en', 'zh', 'fr'])` returns `'en, zh and fr'`
    """
    if len(target_language) ==1:
        return target_language[0]
  
    return ",".join(target_language[:-1])  + ' and ' + target_language[-1]

def upload_image(bucket,image_path, destination_blob_name):
    """
    Uploads an image to a Google Cloud Storage bucket and makes it publicly accessible.

    This function uploads an image to a specified Google Cloud Storage (GCS) bucket. It first attempts to upload the image using the provided `image_path`. 
    If the upload fails, it modifies the `image_path` to remove the `/translated_img/` directory and retries the upload i.e it uplad the raw image from the root if 
    translated path not found.
    The image is made publicly accessible after the upload.

    Returns:
    - str or None: The public URL of the uploaded image if successful, otherwise `None`.

    """
    def upload(blob, path):
        blob.upload_from_filename(path)
        blob.cache_control = "no-cache, no-store, max-age=0"
        blob.patch()
        blob.make_public()
        return blob.public_url
    try:
        blob = bucket.blob(destination_blob_name)
        return upload(blob, image_path)
    except Exception as e:
        save_error(f"Error at upload_image  {e}")
        logger_ocr.error(f"Error in uploading image: {e}")
        # Modify the image path and retry the upload
        modified_image_path = re.sub(r'/translated_img/[^/]+', '', image_path)
        try:
            blob = bucket.blob(destination_blob_name)
            return upload(blob, modified_image_path)
        except Exception as e:
            logger_ocr.error(f"Error in uploading modified image: {e}")
            return None

def upload_to_gcs(bucket_name, source_file_path, destination_blob_folder, credentials_file,epissode_id,index,single=False):
    """
    Initialize the  google cloud storage  and loop through transalted image for uploading.
    """
    # Initialize the Google Cloud Storage client with the credentials
    storage_client = storage.Client.from_service_account_json(credentials_file)
    # Get the target bucket
    bucket = storage_client.bucket(bucket_name)
    url= {}
 
    if single:
        remote_path = os.path.join(destination_blob_folder, source_file_path.split("/")[-1])
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(source_file_path)
        blob.make_public()
        url.append({"index":0,
                    "translated_image_url":  blob.public_url
                    })
        return url
    
    languages = os.listdir(source_file_path)

    urls= []
    print(os.listdir(os.path.join(source_file_path, languages[0])),"Paths    --------")
    for page_number in index:    
        logger_ocr.info(f" Translalted page number {page_number}")
        # print(page_number,"  ----- Translated page number ")
        page_info = {"index": int(page_number), "images": []}
        page_number = str(page_number)+".jpg"
        for language in languages:
            image_path = os.path.join(source_file_path, language, page_number)
            public_url = upload_image(bucket,image_path, f"{epissode_id}/{language}/{page_number}")
            page_info["images"].append({"language": language, "url": public_url})
        urls.append(page_info)

    logger_ocr.info(f" Uploaded urls: {urls}")
    # breakpoint()
    return urls

def specific_instruction(translation_dictionary):
    """
    Generates specific translation instructions based on a provided translation dictionary.

    This function creates a set of instructions for translating specific key phrases in a manga. 
    The instructions are dynamically generated based on the contents of the `translation_dictionary`, detailing how certain key phrases should be 
    translated into multiple target languages.
    Parameters:
        - translation_dictionary (dict): A dictionary where keys are specific phrases found in the manga, and values are dictionaries of translations with language codes as keys.

    Returns:
        - str: A formatted string containing specific instructions for translating the key phrases based on the dictionary.

    The function will generate:
        If ナルト appears on the text of the manga_context value, translate it as "ninja tutule hokage" in english and "佐助" in chinese.

    """    

    output_string = "If {key} appears on the text of the manga_context value, translate it as {translations}."

    # Loop through the dictionary to generate the strings for each key
    final_string = []
    for key, translations in translation_dictionary.items():
        # Create a dynamic list of language-specific translations
        translation_parts = [f'"{translation}" in {lang}' for lang, translation in translations.items()]
        
        # Join the parts with commas and "and" for the last part
        formatted_translations = ', '.join(translation_parts[:-1]) + ' and ' + translation_parts[-1] if len(translation_parts) > 1 else translation_parts[0]
        
        # Format the final output string
        generated_string = output_string.format(
            key=key,
            translations=formatted_translations
        )
        final_string.append(generated_string)

    return ("\n").join(final_string)

def new_simalarcontext_prompt(target_language, manga_name, title, prompt_keys,specific_intructions_prompt):
    """
    Generate the tranlsation prompt.
    
    """
    response_format = {}
    for language in target_language:
        response_format[language] ={}
        for pkeys in prompt_keys:
            response_format[language].update({pkeys: "translated text"})
    target_language = format_target_language(target_language,)
    prompt = f"""I need you to act as a manga translator, tasked with translating Japanese text into {target_language} suitable for a manga audience.
                Your translations should use simple, clear language while preserving the original character names and the tone intended by the original text.
                Use English that fits the tone of a conversation between characters.
                Each line of text given to you must be treated as an individual entity, ensuring that the logical integrity and emotional impact of the original Japanese are maintained in the {target_language} version. Focus on making the translation feel natural and engaging for readers who are familiar with manga culture.
                Avoid complex  words that might detract from the readability and flow suitable for a manga's dialogue and descriptions.
                When translating, consider the space constraints typically found in manga panels and try to keep the text concise yet expressive. 
                You are given an input dictionary with two sections: similar_context and manga_context. Your task is focused solely on manga_context. on manga context
                
                Here is desription of the manga.

                The name of this manga serires is {manga_name} and its chapter  title is {title}.

                Translation Directive:

                Translate the Japanese text within manga_context panels if their is  Japanese text else return empty json. These Japanese text are extracted from the manga. The translation should be semantic, utilizing the similar_context for understanding the narrative and emotional context.

                Specific Translation Instructions:
                Word Selection: Use common words as typically found in manga. Avoid complex terminology.

                Individual Translation: Each text entry in the japanese_text dictionary is a separate entity and must be translated independently.

                Preserve Character Names and Special Terms: Character names (e.g., "All Might") and special terms (eg. Kamehameha, rasengan ) must be kept intact in their translated form. Transcribe these names into {target_language} Unicode but do not integrate them into the surrounding translation such that their distinct identity is lost.

                Preserve Keys: Use the exact keys from the japanese_text dictionary in your output, including any suffixes like '_f'.

                Avoid Merging: Do not combine or overlap translations between keys. Each key represents a distinct piece of dialogue or text and should be translated as such.

                Avoid Switching Keys: Each key in the input must correspond directly to its translation in the output. It is crucial to maintain the integrity of the key-translation pairings throughout the translation process. Do not modify, rearrange, or switch the keys in the output. Each Japanese text entry should be translated and the resulting translation should be assigned back to its original key without any alteration. Ensure that the translated text is linked exactly to the key from which the Japanese text was derived, preserving the structure and order of the input data.
                
                User Reserved Word Directive:

                    
                Technical Considerations:               

                If the text mentions a character name or a special technique, transcribe it into {target_language} Unicode but keep it distinct from other parts of the text.

                Output Specifications: """+ f"""

                Your output should be a dictionary with the same structure as the japanese_text dictionary, where each key corresponds to a translation: the keys should be the target language  i.e {target_language} """ +"""
                """ + str(response_format)+ f"""



            Important Points to Note:
              
                Exclude any translations from the similar_context.

                Ensure each translated segment is self-contained and reflects only its respective key.

                Example for Clarification:

                For the Japanese text keys "0_0" and "0_1":

                "0_0": "あいつは．．．何かを" -> Translate as an independent thought.

                "0_1": "掴みかけている途中だった．．．！" -> Translate as a separate statement.  Do not include symbol like '' on translated text

                Translate text to match the tone of a conversation between characters, maintaining a natural conversational style. If there's a conversational pattern, reflect it in the translation.

                Example for clarification:
                    - Japanese: おう風太郎通りまで送っていってやんな
                    - Translation: Yeah, Futaro, you walk her to the street.


                Remember: The translation for "え～～〜〜〜！！？" is "Ehhhhhh?! What?!". This is an example of handling exclamations or unique expressions
                Return only json format dont return addtional detail 
               
                {specific_intructions_prompt}
        
     """
    return prompt




def reamaining_prompt(prompt,response, remaining_keys ):
    new_prompt= f""" This is the prompt {prompt} \n I got this response from chat gpt api {response} \n 
these are the remaining kesy {remaining_keys} translate remaining keys also and return all keys text on a single json format as mention above.
 
"""
    return new_prompt


def prompt_maker(target_language,prompt_keys,title,manga_name,specific_intructions_prompt):
    """ Generate prompt for translation for page 0 """
    response_format = {}
    for language in target_language:
        response_format[language] ={}
        for pkeys in prompt_keys:
            response_format[language].update({pkeys: "translated text"})
    target_language = format_target_language(target_language,)
    return f"""You are tasked with translating Japanese text from a given dictionary called japanese_text into  target languages: {target_language}.""" +""" Your goal is to ensure that the translations maintain the narrative style and emotional tone of the original content, 
    focusing on contextual meaning over literal translation.

Instructions:
The name of the manga is {manga_name}. The title of this eppisode is {title}
"""+ f"""Process each entry in the japanese_text dictionary and translate the text into given target language i.e  {target_language}"""+""".

Construct a JSON object where each key corresponds to its original Japanese text identifier and the translated texts are returned as values under their respective language keys.

Return translations only for existing keys within japanese_text. If a key under japanese_text contains no actual Japanese text, include an empty JSON object for that key.

Exclude any other manga panel details such as character presence or panel descriptions from the output.

Ensure that the output format aligns with the following structure, with each translated text linked to the original key:

Example Input:

json

{ "title": "the final boss", "number_of_panels": 5, "panel_less": [], "panel_1": { "character_present": "this panel doesn't contains characters", "japanese_text": {"0_0": "何だ！？"} }, "panel_2": { "character_present": "1 number of boy 1 number of girl ", "japanese_text": "this panels doesnt contains text" }, "panel_3": { "character_present": "2 number of boy 2 number of girl ", "japanese_text": {"2_0": "あれ．．．昔敵連合にら致られた．．．", "2_1": "名前何だっけ．．．体育祭で拘束されてた．．．", "2_2": "爆破の〝個性〟：！？"} }, "panel_4": { "character_present": "1 number of boy ", "japanese_text": "this panels doesnt contains text" }, "panel_5": { "character_present": "this panel doesn't contains characters", "japanese_text": "this panels doesnt contains text" } }

Example Output Format:

json 

{""" +  f'{target_language}'+""": { "0_0": "What the!?", "2_0": "That... Back then, targeted by the Villain Alliance...", "2_1": "What was his name again?... He was restrained during the sports festival...", "2_2": "The 'Explosion' Quirk: What!?" }, "chinese": { "0_0": "什么！？", "2_0": "那个...以前被敌人联盟目标...", "2_1": "他的名字是什么来着？... 在体育节上被束缚了...", "2_2": "‘爆炸’个性：什么！？" } }

This format clearly separates each translation by language and adheres to the specified output structure, ensuring clarity and alignment with the input structure. 
Your output should be a dictionary with the same structure as the japanese_text dictionary, where each key corresponds to a translation: the keys should be the target language  i.e {target_language} """ +"""
""" + str(response_format)+ f"""
Important Points to Note:
              
                Exclude any translations from the similar_context.

                Ensure each translated segment is self-contained and reflects only its respective key.

                Example for Clarification:

                For the Japanese text keys "0_0" and "0_1":

                "0_0": "あいつは．．．何かを" -> Translate as an independent thought.

                "0_1": "掴みかけている途中だった．．．！" -> Translate as a separate statement.  Do not include symbol like '' on translated text

                Remember: The translation for "え～～〜〜〜！！？" is "Ehhhhhh?! What?!". This is an example of handling exclamations or unique expressions
                Return only json format dont return addtional detail 
               
                {specific_intructions_prompt}
        
 """


