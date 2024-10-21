import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from manga_ocr import MangaOcr
from translate import Translator
from tqdm import tqdm
from translate_online import deepl, google_translate
import time

# tranlate","deepl","google_translate"
def translate_japanese_to_english(text, trans_medium):
    if trans_medium== "translate":
        translator = Translator(to_lang="en", from_lang="ja") # to_lang="zh-CN" for chinese
        translation = translator.translate(text)
        print(translation)
        return translation
    elif trans_medium == "deepl":
        translated_text = deepl(text)
        return translated_text
      

    elif trans_medium == "google_translate":
        return google_translate(text)


def convert_cv2_image_to_pil(cv2_image):
    # Convert the color space from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert the OpenCV image (NumPy array) to a PIL image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

def add_text_to_image(input_image_path, text):
    # Load the image using OpenCV
    # img = cv2.imread(input_image_path)

    # Get the dimensions of the image
    img_height, img_width, _ = input_image_path.shape

    # Set font and other parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1
    font_color = (0, 0, 0)
    max_line_width = 150

    # Split the text into lines
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        # Check if adding the next word exceeds the maximum width
        if cv2.getTextSize(current_line + ' ' + word, font, font_scale, font_thickness)[0][0] <= max_line_width:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word

    # Add the last line
    lines.append(current_line)

    # Calculate the starting y-coordinate for centering
    total_text_height = len(lines) * 30  # Adjust as needed based on font size and line spacing
    start_y = max((img_height - total_text_height) // 2, 0)

    # Draw each line on the image with justification
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        y = start_y + text_size[1]

        # Calculate the starting x-coordinate for centering
        x = (img_width - text_size[0]) // 2

        # Draw the text with justification
        cv2.putText(input_image_path, line, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        start_y += 20  # Adjust as needed based on font size

    # Convert BGR image to RGB for display
    img_rgb = cv2.cvtColor(input_image_path, cv2.COLOR_BGR2RGB)
    return img_rgb
    # Save the image
    # cv2.imwrite(input_image_path, img_rgb)


def process_image_with_contours(input_image_path,text ):
    # Read the input image
    # image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(input_image_path, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary mask
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask
    mask = np.zeros_like(gray)

    # Draw the largest contour on the mask
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Create a copy of the original image to work on
    result = input_image_path.copy()

    # Fill the region inside the largest contour with smooth white color
    cv2.fillPoly(result, [largest_contour], color=(255, 255, 255))
    added_text = add_text_to_image(result,text)
    return added_text


def main(image_path):
    translation_list = ["translate","deepl","google_translate"]
    image = cv2.imread(image_path)
    output_image = image.copy()    

    detections = text_bubble_detector(image)

    text_bubble_box = detections[0].boxes.data.cpu().numpy()
    destination_img = image_path.split(".")[0]
    medium =""
    translated_data = []
    for i ,box in enumerate(tqdm(text_bubble_box)):
        x1, y1, x2, y2, acc, index = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        if acc >= 0.6:
            text_bubble = output_image[y1:y2,x1:x2]
            original_text = mocr(convert_cv2_image_to_pil(text_bubble))
            translated_data.append([original_text, [x1,y1,x2,y2]])
    for trans_medium in translation_list[:1]:
        output_image = image.copy()    
        for trans_data in translated_data:
            x1, y1, x2, y2 = trans_data[1]
            original_text = trans_data[0]
            text_bubble = output_image[y1:y2,x1:x2]
            translated_text = translate_japanese_to_english(original_text, trans_medium)
            filled_image = process_image_with_contours(text_bubble,translated_text)
            
            output_image[y1:y1 + filled_image.shape[0], x1:x1 + filled_image.shape[1]] = filled_image
            # destination_img 
            medium = trans_medium
        destination_img = image_path.split(".")[0].split("/")[-1] + medium
        cv2.imwrite(f"{destination_img}.png",output_image)
    print(destination_img)
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img",output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import os 
    text_bubble_detector = YOLO('best-detection.pt')
    mocr = MangaOcr()
    # for images in os.listdir("img/one-piece-chapter-1109/"):
    #     image_path = os.path.join("img/one-piece-chapter-1109/", images)
    #     # image_path = "img/shangrila_168/4.jpg"
    #     main(image_path)
    image_path = "img/shangrila_168/2.jpg"
    main(image_path)

    # translate_japanese_to_english("まぁ同じ「巻貝」ってことで","translate")