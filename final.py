from PIL import Image
from ultralytics import YOLO
from manga_ocr import MangaOcr
from translate import Translator
import cv2
import os
import shutil
import glob
import numpy as np
from io import BytesIO
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage


def translate_japanese_to_english(text):
    translator = Translator(to_lang="en", from_lang="ja") # to_lang="zh-CN" for chinese
    translation = translator.translate(text)
    return translation


def add_text_to_image(input_image_path, text):
    # Load the image using OpenCV
    img = cv2.imread(input_image_path)

    # Get the dimensions of the image
    img_height, img_width, _ = img.shape

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
        cv2.putText(img, line, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        start_y += 20  # Adjust as needed based on font size

    # Convert BGR image to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save the image
    cv2.imwrite(input_image_path, img_rgb)



def process_image_with_contours(input_image_path, output_image_path, text):
    # Read the input image
    image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    result = image.copy()

    # Fill the region inside the largest contour with smooth white color
    cv2.fillPoly(result, [largest_contour], color=(255, 255, 255))

    # Save the result
    cv2.imwrite(output_image_path, result)

    # Display the result (optional)
    # cv2.imshow('Smoothly Filled Inside Largest Contour', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Add text to the processed image
    add_text_to_image(output_image_path, text)



def overlay_images(base_image, overlay_image_path, coordinates):
    # Load the overlay image
    overlay_image = cv2.imread(overlay_image_path)

    # Get the dimensions of the overlay image
    height, width, _ = overlay_image.shape

    # Coordinates where overlay_image will be placed on base_image
    x1, y1 = int(coordinates[0]), int(coordinates[1])

    # Calculate the region of interest (ROI) in base_image
    roi_width = min(width, base_image.shape[1] - x1)
    roi_height = min(height, base_image.shape[0] - y1)

    # Overlay overlay_image on base_image within the valid ROI
    if roi_width > 0 and roi_height > 0:
        base_image[y1:y1 + roi_height, x1:x1 + roi_width] = overlay_image[:roi_height, :roi_width]



def display_image(image):
    image_io = BytesIO()
    PILImage.fromarray(image).save(image_io, 'PNG')
    display(IPImage(data=image_io.getvalue(), format='png'))


def clean_up_output_folder(output_folder):
    # Clean up the output folder by deleting all files
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

def clean_up_main_output_image(output_image_path):
    # Clean up the main output image
    if os.path.exists(output_image_path):
        os.remove(output_image_path)

def clean_up_translated_text_images(output_folder):
    # Clean up translated_text_ images
    translated_text_files = glob.glob(os.path.join(output_folder, 'translated_text_*'))
    for file in translated_text_files:
        os.remove(file)

def clean_up_original_text_images(output_folder):
    # Clean up original_text_ images
    original_text_files = glob.glob(os.path.join(output_folder, 'original_text_*'))
    for file in original_text_files:
        os.remove(file)





# Initialize an empty image to accumulate overlays
output_image = None

def process_all_images(input_image_path, output_folder='text_bubbles'):
    global output_image  # Use the global variable

    # Load YOLO model for text bubble detection
    text_bubble_detector = YOLO('best-detection.pt')

    # Clean up the output folder and main output image before running
    clean_up_output_folder(output_folder)
    clean_up_main_output_image('output_image.jpg')
    clean_up_translated_text_images(os.getcwd())
    clean_up_original_text_images(os.getcwd())

    # Open the input image
    input_image = Image.open(input_image_path)

    # Initialize output_image if not already initialized
    if output_image is None:
        output_image = np.array(input_image)

    # Perform text bubble detection
    detections = text_bubble_detector(input_image)
    text_bubble_boxes = detections[0].boxes.data.cpu().numpy()

    # Save and display all text bubbles at once
    for i, box in enumerate(text_bubble_boxes):
        x1, y1, x2, y2, _, _ = box
        text_bubble_filename = os.path.join(output_folder, f'text_bubble_{i}.jpg')

        # Crop and save each text bubble
        text_bubble = input_image.crop((x1, y1, x2, y2))
        text_bubble.save(text_bubble_filename)

    # Display all text bubbles at once
    for i, box in enumerate(text_bubble_boxes):
        x1, y1, x2, y2, _, _ = box
        text_bubble_filename = os.path.join(output_folder, f'text_bubble_{i}.jpg')

        # Translate Japanese text to English
        mocr = MangaOcr()
        original_text = mocr(text_bubble_filename)
        translated_text = translate_japanese_to_english(original_text)

        # Display the original and translated text bubbles separately
        plt.figure(figsize=(15, 5 * len(text_bubble_boxes)))

        plt.subplot(len(text_bubble_boxes), 2, 2 * i + 1)
        img = Image.open(text_bubble_filename)
        plt.imshow(img)
        plt.title(f'Text Bubble {i + 1}\nOriginal Text: {original_text}, Translated Text: {translated_text}')

        # plt.subplot(len(text_bubble_boxes), 2, 2 * i + 2)
        # plt.imshow(img)
        # plt.title(f'Translated Text: {translated_text}')

    plt.show()

    # Process user input for each text bubble
    for i, box in enumerate(text_bubble_boxes):
        x1, y1, x2, y2, _, _ = box
        text_bubble_filename = os.path.join(output_folder, f'text_bubble_{i}.jpg')
    
        # Translate Japanese text to English
        mocr = MangaOcr()
        original_text = mocr(text_bubble_filename)
        translated_text = translate_japanese_to_english(original_text)
    
        # Display the original and translated text
        print(f'Coordinates: {x1}, {y1}, {x2}, {y2}')
        print(f'Text Bubble {i + 1} - Original Text: {original_text}')
        print(f'Text Bubble {i + 1} - Translated Text: {translated_text}')
    
        # Prompt the user to choose whether to keep the original or overlay the translated text
        while True:
            try:
                user_choice = input(f"Do you want to keep the original text bubble {i+1}? (yes/no): ").lower()
                if user_choice in ['yes', 'no']:
                    break
                else:
                    raise ValueError("Invalid input. Please enter 'yes' or 'no'.")
            except ValueError as e:
                print(e)
    
        if user_choice == 'no':
            # Process the image with contours and overlay the translated text
            output_image_path = f'translated_text_{i}.jpg'
            process_image_with_contours(text_bubble_filename, output_image_path, translated_text)
    
            # Overlay the translated text onto the original image
            overlay_images(output_image, output_image_path, (x1, y1))
        else:
            # Keep the original text bubble
            output_image_path = f'original_text_{i}.jpg'
            process_image_with_contours(text_bubble_filename, output_image_path, original_text)

    # Ask for confirmation before saving the final image
    while True:
        try:
            user_confirmation = input("Do you want to save the final image with all changes? (yes/no): ").lower()
            if user_confirmation in ['yes', 'no']:
                break
            else:
                raise ValueError("Invalid input. Please enter 'yes' or 'no'.")
        except ValueError as e:
            print(e)

    if user_confirmation == 'yes':
        # Save the final result
        cv2.imwrite('output_image.jpg', output_image)
        print("Final image saved successfully!")
    else:
        print("Changes not saved. Exiting without saving.")

    # Display the final result
    display_image(output_image)




if __name__ == "__main__":
    # Specify the input image path
    input_image_path = 'img/17.jpg'

    # Perform text bubble detection and processing for all images
    process_all_images(input_image_path)

    

