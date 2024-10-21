import cv2
import os
import numpy as np
import weaviate
from manga_ocr import MangaOcr
from ultralytics import YOLO
import tensorflow.keras.backend as K
import utils as ut
# from yolo_app import YOLOApp
from text_overlay import ChineseTextOverlay, EnglishTextOverlay
from utils import sort_panels_by_reading_order, fill_text_bubble,assign_to_panel, sort_text_bubble_by_reading_order, parase_character, get_text,handel_panel_less,translate_chatgpt
from config import OPEN_AI
import config as cfg
import concurrent.futures
import time
max_workers=4



class DetectedObject:
    """
    Store the co-oridinates  and other meta data of the detected text bubble and gender detected on the manga page
    """
    def __init__(self, detection_box,gender=None,text_cooordinate=None):
        self.x = int(detection_box[0])
        self.y = int(detection_box[1])
        self.width = int(detection_box[2])
        self.height = int(detection_box[3])
        self.acc=  detection_box[4]
        self.cls_index=  detection_box[5]
        self.panel_index = None  
        self.translated_text = None 
        self.text_color= None
        self.gender = gender
        self.text_cooordinate = text_cooordinate
        self.bubble_shape = ""

class MangaPageAnalyzer:
    """
    Class responsible for analyzing manga pages, detecting objects, extracting text, and overlaying translations.

    The `MangaPageAnalyzer` class handles the complete workflow for processing manga pages. It detects panels, text bubbles, and 
    characters within a manga image, extracts and translates text, and overlays the translated text back onto the image. 
    The class utilizes multiple models, including an object detection model, OCR model, and gender classification model, to perform these tasks.

    Attributes:
        - model: Object detection model used to identify text bubbles, characters, and panels.
        - mocr: OCR model for extracting text from detected text bubbles.
        - image_path (str): File path of the manga image being analyzed.
        - gender_model: Model for classifying the gender of detected characters.
        - image: The manga image loaded from the specified `image_path`.
        - detections (list): List to store detected objects within the image.
        - clases_index (list): List to store indices of detected classes.
        - panels (list): List to store detected panels within the image.
        - text_bubble (list): List to store detected text bubbles within the image.
        - characters (list): List to store detected characters within the image.
        - text_id (dict): Dictionary mapping detected text to unique identifiers.
        - bubble_shape (str): Shape of the detected speech bubbles.
        - prompt (str): Generated prompt based on detected text and panels.
        - text_contains (str): Indicates whether the page contains translatable text.
        - story_prompt (str): Prompt used for generating the story or translation based on the page content.

    """
    def __init__(self, model, mocr,image_path,gender_model):
        self.model = model
        self.mocr = mocr
        self.image_path = image_path
        self.gender_model= gender_model
        self.image = cv2.imread(image_path)
        self.detections = None
        self.clases_index = None
        self.panels = []
        self.text_bubble = []
        self.characters = []
        self.text_id= {}
        self.bubble_shape = ""
        self.prompt=""
        self.text_contains=""
        self.story_prompt=""
       

    def get_gender(self,box):
        """
        Classifies the gender of a detected character based on the bounding box.
        """
        crop_img = self.image[box[1]:box[3],box[0]:box[2]]
        crop_img = cv2.resize(crop_img,(120,120))
        expand = np.expand_dims(crop_img,axis=0)
        predictions = self.gender_model.predict(expand,verbose=False)
        if K.sigmoid(predictions[0][0]) < 0.5:
            return "girl"
        else:
            return "boy"
    
    def custom_nms(self,predictions, iou_threshold=0.5):
        """
        Performs Non-Maximum Suppression (NMS) on detection results to remove duplicate detections.
        """
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
            
            ious = np.array([self.iou(current_box, other_box) for other_box in other_boxes])
            
            indices = indices[1:][ious <= iou_threshold]
        
        return predictions[keep_boxes]
    
    def iou(self,box1, box2):
        """
        Computes the Intersection over Union (IoU) for two bounding boxes.
        """
        x1, y1 = np.maximum(box1[:2], box2[:2])
        x2, y2 = np.minimum(box1[2:4], box2[2:4])
        
        inter_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
        
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        
        iou = inter_area / (box1_area + box2_area - inter_area)
        
        return iou
    
    def detect_objects(self):
        """
        Detects panels, text bubbles, and characters within the manga image and stores them in the corresponding attributes.
        """
        # self.image_blur = cv2.blur(self.image,(3,3))
        self.detections  = self.model.predict(self.image,iou=0.5, conf=0.5) #0.6 optimal accuracy 
        
        self.clases_index = self.detections[0].names
        box  = self.detections[0].boxes.data.cpu().numpy()
        box = self.custom_nms(box,iou_threshold=0.5)
        # print(len(box), "*"*10)
        for panel in box:
            if panel[-1] == 2:
                self.panels.append(DetectedObject(panel))
            elif (panel[-1] == 0 and panel[-2] >=0.5 )or (panel[-1] == 1 and panel[-2] >=0.76) :
                self.text_bubble.append(DetectedObject(panel))
                
            else:
                box = [int(bbox) for bbox in panel[:4]]
                gender = self.get_gender(box)
                self.characters.append(DetectedObject(panel,gender))
        return self.image

    def analyze_page(self):
        """
        Sorts detected panels by reading order, assigns detected text bubbles and characters to their respective panels, and handles any text or characters that are not associated with a specific panel.
        """
        self.panels = sort_panels_by_reading_order(self.panels)
        assign_to_panel(self.text_bubble + self.characters, self.panels)
        panless_text = [text for text in self.text_bubble if text.panel_index is None]
        panless_character = [text for text in self.characters if text.panel_index is None]
        handel_panel_less(panless_text + panless_character, self.panels)

    
    def process_single_panel(self, panel_id: int):
        """
        Processes a single panel by extracting text and character information, generating a prompt for translation.

        """
        texts = [obj for obj in self.text_bubble if obj.panel_index == panel_id]
        characters = [obj for obj in self.characters if obj.panel_index == panel_id]
        texts = sort_text_bubble_by_reading_order(texts)
        characters_text = parase_character(characters, self.clases_index)
        japanese_text = get_text(texts, self.image, panel_id, self.text_id, self.mocr)
        japanese_text_present = japanese_text if japanese_text else 'this panel doesn’t contain text'
        characters_present = str(characters_text) if characters_text else "in this characters are not detected by the model chracters may presents"

        panel_data = {
            "character_present": characters_present,
            "japanese_text": japanese_text_present
        }
        story_text = f" - Panel {panel_id+1} contains {characters_text} and the following text: {japanese_text}\n"
        return panel_id, panel_data, story_text
    
    def process_panels(self, title):
        """
        Processes all panels on the page, generating prompts and a story prompt for translation.
        """
        # draw_on_image(self.image, self.panels, self.clases_index, "panel")
        # draw_on_image(self.image, self.text_bubble, self.clases_index, "text_bubble")

             
        prompt ={
                "title": title,
                "number_of_panels": len(self.panels),
                "panel_less":[],
                "only_text":[]

            }
        story_prompt = f"""You will be provided the japanese text of manga with it associated panel and associated characters on that panel. 
        Task is to create a viable story line from the Japanese text. This story will be used  as  a context for the translation of japanese text get semantic translation rather than literal meaning. 
        The chat gpt api will be used for translation.This manga page is divided into {len(self.panels)} panels. The title of the page is {title}. Here is the japanese text:"""
        story_user_prompt = ""
        if len(self.panels) >0 :
             with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_single_panel, id): id for id in range(len(self.panels))}
                for future in concurrent.futures.as_completed(futures):
                    panel_id = futures[future]
                    try:
                        panel_id, panel_data, story_text = future.result()
                        prompt[f"panel_{panel_id+1}"] = panel_data
                        story_user_prompt += story_text
                    except Exception as exc:
                        print(f"Panel {panel_id} generated an exception: {exc}")
                        ut.save_error(f"Error at process_panels: {exc}")
            
        else:
            # print("panel less text", self.text_bubble)
            id = 0
            characters_text = parase_character(self.characters,self.clases_index)
            japnese_text = get_text(self.text_bubble,self.image,"0_P",self.text_id, self.mocr)
            if len(characters_text)>0:
                prompt["panel_less"].append({"character_present": characters_text,
                                             "japanese_text": japnese_text})
                story_user_prompt += f" - Panle {id+1} contains {characters_text } and following text {japnese_text} \n"
                id +=1 
            else :
                prompt["only_text"].append(japnese_text)
                story_user_prompt += f"This page contains no character but contains followinf text {japnese_text}"

                
        story_user_prompt += "generate the overall story doesn't need to associates panels in story \
                        return the summarized story in json format {'story':summarized story}. Genereate story if only japanese text present esle return this {'story':''}  "
        
        text_contains = True if len(self.text_bubble) >0 else False
        self.prompt = prompt
        self.story_prompt = story_prompt
        prompt = dict(sorted(prompt.items()))
        return prompt,story_prompt,story_user_prompt, text_contains
    
    def adjust_text_box(self,bounding_box, segmented_text_box):
        """
        Adjusts the bounding box for segmented text within a panel.
        """
        x1, y1, x2, y2 = bounding_box
        sx1, sy1, sx2, sy2 = segmented_text_box

        # Calculate the height of the bounding box
        bounding_box_height = y2 - y1

        # Calculate 20% of the bounding box's height
        threshold = 0.2 * bounding_box_height

        # Check if the start point of the segmented box is greater than 20% of the bounding box size
        if sy1 - y1 > threshold:
            # Decrease the start point of the segmented text box's Y-coordinate
            # print("distance  ", sx1,y1, bounding_box_height, threshold)
            sy1 = y1 + int(threshold)

        # Return the adjusted segmented text box
        return sx1, sy1, sx2, sy2
    
    def translate_and_draw(self, response, image_save_path="",img_save=True):
        """
        This method processes the response containing translated text for detected text bubbles, adjusts the text boxes, 
        and overlays the translated text onto the manga image. The final image can either be saved to the specified directory or displayed.

        Parameters:
            - response (dict): A dictionary containing the translated text, organized by language and text bubble identifiers.
            - image_save_path (str, optional): The directory path where the translated images should be saved. Default is an empty string.
            - img_save (bool, optional): Flag indicating whether to save the image or display it. Default is `True`.

        Workflow:
            1. Creates a copy of the original manga image for processing (`output_image`).
            2. Iterates over each language and the corresponding translated text in the `response`.
            3. For each detected text bubble:
            - Retrieves the bounding box and related information.
            - Uses the `fill_text_bubble` function to remove the original text and prepare the bubble for the translated text.
            - Stores the color and bubble shape for later use.
            4. In the second loop:
            - For each language, creates a new image copy for overlaying the translated text.
            - Calculates the new bounding box and adjusts it based on the segmented text box.
            - Depending on the language, either a Chinese or English overlay is applied using `ChineseTextOverlay` or `EnglishTextOverlay`.
            5. Saves the final image to the specified path or displays it, depending on the `img_save` flag.
            6. Handles any exceptions during the process and logs errors.

        Exceptions:
            - Catches and logs any exceptions that occur during the translation or overlay process, ensuring robustness in handling errors.

        Returns:
            - None: The method does not return any value but either saves or displays the processed image.
        """

        output_image = self.image.copy()
         
        for tlangauge, tvalue in response.items():
            try:
                for key,value in tvalue.items():
                    self.text_id[key][0].translated_text = value
                    # x,y,w,h = int(self.text_id[key][0].text_cooordinate[0]),int(self.text_id[key][0].text_cooordinate[1]),int(self.text_id[key][0].text_cooordinate[2]),int(self.text_id[key][0].text_cooordinate[3])
                    x,y,w,h = int(self.text_id[key][0].x),int(self.text_id[key][0].y),int(self.text_id[key][0].width),int(self.text_id[key][0].height)
                    
                    text_bubble = output_image[y:h,x:w]
                    # cv2.imwrite(f"img/translated_img/tokyo/{key}.jpg", text_bubble)
                    filled_image, text_color, self.text_id[key][0].bubble_shape= fill_text_bubble(text_bubble,key)
                    # print("Buuble shape :    ----------- ", self.text_id[key][0].bubble_shape ," ----------")
                   
                    output_image[y:y + filled_image.shape[0], x:x + filled_image.shape[1]] = filled_image  
                    # data_for_app.append({"id":key,"bbox":[x,y,w,h], "text":value})
                    self.text_id[key][0].text_color = text_color
                break
            except Exception as e:
                ut.save_error(e)
                print( " exception at removing text:::   ", e )
                pass
          
        
        # print("Final Response::   ", response)
        try:
            for tlangauge, tvalue in response.items():
                temp_img = output_image.copy()
                # font_path =os.path.join(cfg.FONT_PATH,f"{tlangauge}/font.ttf")
                # padding = False
            # print(font_path)
                for key, value in tvalue.items():
                    # new_x = self.text_id[key][0].text_cooordinate[0]- (self.text_id[key][0].text_cooordinate[0]-self.text_id[key][0].x  )//2

                    # new_x2 = self.text_id[key][0].text_cooordinate[2]+ (self.text_id[key][0].width - self.text_id[key][0].text_cooordinate[2]) //2

                    # # start_y =  self.text_id[key][0].text_cooordinate[1] - self.text_id[key][0].y

                    # if new_x == self.text_id[key][0].x  :
                    #     padding= True
                    # else:
                    #     padding = False
                    x,y,w,h = int(self.text_id[key][0].x),int(self.text_id[key][0].y),int(self.text_id[key][0].width),int(self.text_id[key][0].height)
                    x2,y2,w2,h2 = self.text_id[key][0].text_cooordinate[0],self.text_id[key][0].text_cooordinate[1],\
                                    self.text_id[key][0].text_cooordinate[2],self.text_id[key][0].text_cooordinate[3]
                    x,y,w,h = ut.calculate_new_bounding_box([x,y,w,h],(x2,y2,w2,h2),distance_ratio=0.45)

                    # x,y,w,h = int(new_x),int(self.text_id[key][0].text_cooordinate[1]),int(new_x2),int(self.text_id[key][0].text_cooordinate[3])
                    # print("text     ", value, [x,y,w,h])
                    # x,y,w,h = self.adjust_text_box(bounding_box=[self.text_id[key][0].x,self.text_id[key][0].y,self.text_id[key][0].width,self.text_id[key][0].height],
                    #                                segmented_text_box= [x,y,w,h] )
                    
                    text_bubble = temp_img[y:h,x:w]
                    
                    bubble_shape = self.text_id[key][0].bubble_shape

                    if tlangauge == "zh":
                        chinese_image = ChineseTextOverlay(img=temp_img,
                                                           translated_text=value,
                                                           key=key,
                                                           color=self.text_id[key][0].text_color,
                                                           box=[x,y,w,h])
                        temp_img = chinese_image.put_text_inside()
                    elif tlangauge == "en":
                        englsh_image = EnglishTextOverlay(img=temp_img,
                                                           translated_text=value,
                                                           key=key,
                                                           bubble_shape=bubble_shape,
                                                           color=self.text_id[key][0].text_color,
                                                           box=[x,y,w,h])

                    
                        temp_img = englsh_image.put_text_inside(padding_status=False)
        
                
                index= self.image_path.split("/")[-1].split(".jpg")[0]+".jpg"
                if img_save:
                    # print(f"Image save path :::   {image_save_path}/{tlangauge.lower()}/{index}")
                    cv2.imwrite(f"{image_save_path}/{tlangauge.lower()}/{index}", temp_img)
                else:
                    # app = YOLOApp(data_for_app,temp_img)
                    # app.mainloop()
                    # # print(data_for_app)
                    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
                    cv2.imshow("output",temp_img)
                    print(image_save_path," path ----------")
                    cv2.imwrite(image_save_path, temp_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        except Exception as e:
            print("Error at adding text on image :  ",e)
            print(response)
            ut.save_error(e)
            raise
    


if __name__ == "__main__":
    import time
    from config import OPEN_AI
    import ast
    import os
    import json
    from openai import OpenAI
    import keras
    import config as cfg
    from natsort import natsorted

    client = OpenAI(api_key = OPEN_AI)




    weaviet_client =  weaviate.Client(url=cfg.WEAVIET_URL,
                        additional_headers={ "X-OpenAi-Api-Key": OPEN_AI}
                    )
    print('Weaviet client connnected :', weaviet_client.is_ready())
    ut.get_weaviet_schema(weaviet_client=weaviet_client)
    model_path = cfg.MODEL_PATH
    
    model = YOLO(model_path)
    mocr = MangaOcr()
    with open('model/gender/config.json', 'r') as f:
        model_config = f.read()

    gender_model = keras.models.model_from_json(model_config) 
    gender_model.load_weights ("model/gender/model.weights.h5")
    # for images in os.listdir("/home/mrcloud/Documents/projects/manga_download/mangas/test/"):
    #     if ".jpg" in images:
            # image_path = os.path.join("/home/mrcloud/Documents/projects/manga_download/mangas/test/", images)
    start_time = time.time()
    error_img = ["219","222","237","245","249","270","279","282","283","285","287"]
    folder_path = "img/test/"
    translation_dictionary =  {
        "ナルト": {
            "en": "ninja tutule crow",
            "zh": "佐助"
        },
        "悪りぃ": {
            "en": "thankyou",
            "zh": "谢谢"
        },
        "ジョイボーイ": {
            "en":"Sad boy",
            "zh": "悲情男孩"
        },
        "敵": {
            "en":"best firend",
            "zh":"至交"
        }
    }
    if translation_dictionary:
        specific_intructions_prompt = "\nSpecific Custom Instructions:\n"+ ut.specific_instruction(translation_dictionary=translation_dictionary)
        # specific_intructions_prompt += "\n Fuutarou is the name of the character"
    else:
        specific_intructions_prompt=""
    for  img_page in os.listdir(folder_path):
    
        # print(img_page)
        image_name =f"{0}"

        image_path =f"/home/mrcloud/Downloads/[赤坂アカ] かぐや様は告らせたい～天才たちの恋愛頭脳戦～ 第01巻/0003.jpg"
        # image_path = os.path.join(folder_path, img_page)
        # image_name= img_page.split(".jpg")[0]
        analyzer = MangaPageAnalyzer(model, mocr,image_path, gender_model)
        analyzer.detect_objects()
        analyzer.analyze_page()
        prompt,story_prompt,story_user_prompt, text_contains  = analyzer.process_panels(title="the final boss")

        story = translate_chatgpt(story_prompt,story_user_prompt, client)
        story = ut.parser_and_validate_story_prompt(response=story,client=client)

        target_language= ["en"]
        manga_name = "Bokuno hero acedemia"
        title = "final boss"

        story_data = {
            "manga_name":"ajin",
            "title": "ajin  Q (New Edition)",
            "story" : story["story"],
            "page_no": image_name,
            "context": prompt,
            "chapter": "63"
        }

        # print(story_data ,' Story printed')
        ut.add_record_to_weaviet(weaviet_client= weaviet_client,story_data=story_data)
        similar_context = ut.get_similar_context(weaviet_client=weaviet_client, story_data= story_data)
        # print(similar_context)
        similar_prompt = similar_context
        prompt_dict = {
            "similar_context": similar_prompt,
            "manga_context": prompt
        }

        # quit()
        if text_contains:
            prompt_keys =list( ut.extract_japanese_text(prompt_dict["manga_context"]).keys())
            end_prompt = f" ``` Note: only trasnlate these keys text {prompt_keys} return all the tranlated text belongs to this keys"
            
            # target_language = ut.format_target_language(langauge)
            system_prompt = ut.new_simalarcontext_prompt(target_language=target_language,
                                            manga_name=manga_name,
                                            title=title,
                                                prompt_keys = prompt_keys,
                                                specific_intructions_prompt=specific_intructions_prompt
                                            ) 
            user_prompt = str(prompt_dict) + end_prompt
            # print(prompt, "\n *********")

            response = translate_chatgpt(system_prompt,user_prompt,client)
            response = ut.parse_and_validate_context (response=response,client=client, target_language=target_language, prompt_keys=prompt_keys)

            res_keys = list(response.keys())[0]
        
            remaining_keys = set(prompt_keys) - set(response[res_keys].keys())
            if remaining_keys:
                new_prompt = ut.reamaining_prompt(prompt=user_prompt,
                                            response=response,
                                            remaining_keys= remaining_keys)
                
                remaining_response = translate_chatgpt(new_prompt, client)
                response = {**response, **remaining_response}
     
        else :
            response ={}
        image_save_path = f"img/translated_img/4840000010619/{image_name}.jpg"
        analyzer.translate_and_draw(response,image_save_path=image_save_path, img_save=False)
        break



    print((time.time()-start_time)/60,time.time()-start_time,  "End time  ------" )
