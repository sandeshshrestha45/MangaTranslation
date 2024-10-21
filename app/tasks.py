from fastapi import HTTPException
import requests
import os
import shutil
from manga_ocr import MangaOcr
from ultralytics import YOLO
from pathlib import Path
import asyncio
import cv2
from natsort import natsorted
import httpx
import multiprocessing
import weaviate
import keras
import gc
import time
from celery import signals
from celery_worker import celery
from openai import OpenAI
import config as cfg
from google.cloud import storage
from manga_page_analyzer import MangaPageAnalyzer
from utils import (translate_chatgpt,
                   get_weaviet_schema,
                   add_record_to_weaviet,
                    get_similar_context,
                    save_error,
                    extract_japanese_text,
                    delete_folder,
                    reamaining_prompt,
                    parse_and_validate_context,
                    parser_and_validate_story_prompt,
                    new_simalarcontext_prompt,
                    upload_image,
                    specific_instruction
                     )
from config import OPEN_AI, logger_ocr
from concurrent.futures import  ThreadPoolExecutor

mocr =None
open_api_client = OpenAI(api_key = OPEN_AI)
model = None
weaviet_client= None
gender_model = None
executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2)
max_pool = multiprocessing.cpu_count() -1


@signals.worker_process_init.connect
def load_resources(**kwarg):
    global mocr, model, weaviet_client, open_api_client, gender_model
    mocr = MangaOcr()
    model = YOLO(cfg.MODEL_PATH)
    open_api_client = OpenAI(api_key = OPEN_AI)
    
    
    weaviet_client =  weaviate.Client(url=cfg.WEAVIET_URL,
                            additional_headers={ "X-OpenAi-Api-Key": OPEN_AI},\
                            timeout_config=(10, 120)
                        )
    
    with open(cfg.GENDER_MODEL_CONFIG, 'r') as f:
        model_config = f.read()

    gender_model = keras.models.model_from_json(model_config) 
    gender_model.load_weights (cfg.GENDER_MODEL)
    logger_ocr.info(f'Weaviet client connnected : {weaviet_client.is_ready()} ')
    logger_ocr.info(f'*'*60+"\n")
    get_weaviet_schema(weaviet_client=weaviet_client)


def generate_prompt(image_paths,title):
    """
    Generates translation prompts for each manga image.

    This function processes a list of image paths, analyzes each image using a `MangaPageAnalyzer`, and generates translation prompts based on the content. The prompts are collected and returned for further processing.

    Parameters:
        - image_paths (list): A list of file paths for the images to be processed.
        - title (str): The title of the manga chapter or episode, used as part of the prompt generation.

    Workflow:
        1. Checks if the necessary resources (`model`, `mocr`, `gender_model`) are loaded, and loads them if they are not.
        2. Iterates over the provided `image_paths`.
        3. For each image, initializes a `MangaPageAnalyzer` with the loaded models and the image path.
        4. Detects objects in the image and analyzes the page layout.
        5. Generates translation prompts, story prompts, and text content from the analyzed panels.
        6. Appends the generated prompts along with the analyzer object to the `prompts` list.
        7. Logs any errors encountered during prompt generation.

    Returns:
        - prompts (list): A list of dictionaries, where each dictionary contains the image path and its associated prompts (`prompt`, `story_prompt`, `text_contains`) and the `analyzer` object for further use.
    """

    global model, mocr, gender_model
    if model == None:
        load_resources()
    prompts = []
    for img_path in image_paths:
        try:
            analyzer = MangaPageAnalyzer(model=model, mocr=mocr, image_path= img_path, gender_model=gender_model)
            analyzer.detect_objects()
            analyzer.analyze_page()
            prompt, story_prompt,story_user_prompt,text_contains = analyzer.process_panels(title)
            prompts.append({img_path:[prompt, story_prompt, text_contains,analyzer,story_user_prompt]})
        except Exception as e:
            save_error(f"Error at generate prompt: {e} ")
            logger_ocr.error(f"Error at generate prompt : {e} ------------------------")
    return prompts

def download_image (image_url:str, folder_path,priority):
    logger_ocr.info(image_url)
    try:
        response = requests.get(image_url,stream=True)
        response.raise_for_status() 
        if response.status_code == 200:
            file_path = os.path.join(folder_path, f"{priority}.jpg")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        save_error(f"Error at download_image {e}")
        logger_ocr.error(f"Error in downloading image url: {image_url} --- {e}")



def download_img(image_data, folder_path):
    print(f"Image data \n{image_data}")
    # for img_data in image_data:
    url = image_data["image_url"]
    priority = image_data["index"]
    print(f"Image url: {priority} {url}")
    file_path=download_image(url,folder_path,priority)
    if file_path:
        return "image downloaded"

@celery.task
def download_image_parallel(image_data, folder_path):
    """
    Celery task to download manga images in parallel using a thread pool.

    This function handles downloading multiple manga images concurrently by utilizing a `ThreadPoolExecutor`. The images are fetched based on the provided `image_data` and saved to the specified `folder_path`. The task is optimized for faster downloads by processing multiple images simultaneously.

    Parameters:
        - image_data (list): A list of dictionaries, each containing information about the image, such as the image URL and index.
        - folder_path (str): The directory path where the downloaded images will be saved.

    Workflow:
        1. A thread pool is created with a specified number of worker threads (`max_pool`).
        2. The `executor.map` method is used to apply the `download_img` function to each image in `image_data`, passing the corresponding `folder_path`.
        3. The results of the download operations are collected and returned as a list.

    Returns:
        - list: A list of results from the image download operations, typically indicating the success or failure of each download.
    """

    # print(image_data, type(image_data))
    with ThreadPoolExecutor(max_workers= max_pool) as executor:
        results = executor.map(download_img,image_data,[folder_path]* len(image_data))
    return list(results)



@celery.task
def generate_prompt_parallel(batch_path, title):
    # with ProcessPoolExecutor(max_workers=3) as executor:
    future_results = [executor.submit(generate_prompt,img_path, title ) for img_path in batch_path]

    results = [future.result() for future in future_results]

    return results


def draw_on_image(prompts):
    for img_data in prompts:
        for key, value in img_data.items():
            img = cv2.imread(key)
            bounding_box = value[1]
            for k, v in bounding_box.items():
                # print(bounding_box)
                # for bbox in bounding_box:
                x,y,w,h =int( v[0].x), int(v[0].y),int(v[0].width), int(v[0].height)
                cv2.rectangle(img,(x,y),(w,h),(255,0,0),1)

                # print(x,y,w,h)
                cv2.imwrite(f"{key.split('/')[-1]}", img)

def generate_story(prompt,image_data,extracted_datas):
    """
    Generates and processes story translations for a given prompt.

    This function takes a prompt, generates a story translation using an external translation service (e.g., ChatGPT), and then processes and validates the translated story. The function structures the translation data and appends it to a list of extracted data for further processing.

    Parameters:
        - prompt (dict): A dictionary containing the image path as the key and associated prompt data as the value.
        - image_data (dict): The JSON object containing details about the manga, including series title, chapter ID, and other metadata.
        - extracted_datas (list): A list to collect and store the extracted and processed story data.

    Workflow:
        1. Iterates over the items in the `prompt` dictionary.
        2. Extracts relevant data, such as the prompt, story prompt, and other metadata, from the values associated with the image path.
        3. Generates a story translation by calling the `translate_chatgpt` function with the story prompt.
        4. Validates and parses the generated story using the `parser_and_validate_story_prompt` function.
        5. Structures the translated story data, including manga name, title, chapter, story content, page number, and prompt context.
        6. Appends the structured data to the `extracted_datas` list.
        7. Logs the processed page number for tracking purposes.
        8. Adds the processed story data to a vector database using `add_record_to_weaviet`.

    Returns:
        - extracted_datas (list): The updated list containing the processed and structured story data for each image prompt.
    """

    for key, values in prompt.items():
        prompt = values[0]
        # text_id = values[1]
        # image = values[2]
        image_name = key.split("/")[-1]
        page_no = key.split("/")[-1].split(".")[0]
        story_prompt= values[1]
        story_user_prompt = values[4]
        text_contains= values[2]
        
        story = translate_chatgpt(system_prompt=story_prompt,user_prompt=story_user_prompt, client=open_api_client)
        story = parser_and_validate_story_prompt(response=story,client=open_api_client)

        story_data = {
            "manga_name": image_data["series_title"].lower(),
            "title":image_data["title"].lower(),
            "chapter":str(image_data["id"]),
            "story" : story["story"].lower(),
            "page_no":page_no,
            "context": str(prompt)
        }
        extracted_datas.append({key:{
            # "text_id": text_id,
            # "image": image,
            "image_name":  image_name,
            "text_contains": text_contains,
            "story_data": story_data,
            "prompt": prompt,
            "object": values[-2]

        }})
        logger_ocr.info (f" page No -------------- {page_no}")
        add_record_to_weaviet(weaviet_client= weaviet_client,story_data=story_data)

    return extracted_datas


def translate_image_and_draw(extracted_data,target_language, translated_path,image_data,specific_intructions_prompt):
    """
    Translates and draws text on a manga image based on the extracted data.

    This function processes a single manga image by translating the text found within it and then drawing the translated text onto the image. 
    The function handles both images with text (where translation and drawing are required) and those without text (which are simply copied to the output folder). 
    The processed images are saved in the specified `translated_path`.

    Parameters:
        - extracted_data (dict): A dictionary containing the extracted data for the image, including the prompt, story data, image path, and associated metadata.
        - target_language (list): A list of target languages (e.g., ['en'] for English) for the translation.
        - translated_path (str): The directory path where the translated images will be saved.
        - image_data (dict): The JSON object containing details about the manga, including series title, chapter ID, and other metadata.
        - specific_intructions_prompt (str): A string containing any custom translation instructions that should be applied.

    Workflow:
        1. Iterates over the `extracted_data` items to process each image.
        2. Checks if the image contains text that needs translation (`text_contains`).
        3. If no text is found, the image is copied directly to the target directory for each target language.
        4. If text is found:
            - For the first page (`page_no == 0`), generates a prompt for translation and calls the translation service.
            - For other pages, retrieves similar context from a vector database and creates a detailed prompt.
            - Calls the translation service with the generated prompt and processes the response to ensure all text keys are translated.
        5. Uses the `translate_and_draw` method of the `object` in `values` to draw the translated text onto the image and save it in the target directory.
        6. Handles exceptions during translation and logs any errors encountered.

    Returns:
        - None: The function performs translation and saves the images, but does not return any value.
    """

    for key, values in extracted_data.items():
        text_contains = values["text_contains"]
        story_data = values["story_data"]
        prompt = values["prompt"]
        image_name = values["image_name"]
  
        page_no = key.split("/")[-1].split(".")[0]
        if not text_contains:
        
            for tlanguage in target_language:
                shutil.copy(key,os.path.join(translated_path,f"{tlanguage}/{image_name}") )
        else:
            try:
                if int(page_no ) <= 1:
                     # extract all the keys of the prompt that is needed to be translate
                    prompt_keys =list( extract_japanese_text(prompt).keys())
                    
                    prompt_dict = {
                        "similar_context": {},
                        "manga_context": prompt
                    }
                else:
                    simlar_context = get_similar_context(weaviet_client=weaviet_client,
                                                        story_data=story_data)
                    
                    prompt_dict = {
                        "similar_context": simlar_context,
                        "manga_context": prompt
                    }
                prompt_keys =list( extract_japanese_text(prompt_dict["manga_context"]).keys())
                end_prompt=  f" ``` Note: only translate these keys text {prompt_keys} return all the translate text belongs to this keys"
                system_prompt = new_simalarcontext_prompt(target_language=target_language,
                                                manga_name= image_data["series_title"].lower(),
                                                title=image_data["title"].lower(),
                                                prompt_keys=prompt_keys,
                                                specific_intructions_prompt=specific_intructions_prompt
                                                )
                
                user_prompt = str(prompt_dict) + end_prompt

                # sending prompt to chat gpt for translation 
                response = translate_chatgpt(system_prompt=system_prompt, user_prompt=user_prompt, client=open_api_client)

                # insuring response is on desired format
                response = parse_and_validate_context (response=response,client=open_api_client, target_language=target_language, prompt_keys=prompt_keys)
                logger_ocr.info(response)
                if response:
                    res_keys = list(response.keys())[0]
                    # breakpoint()

                    # checking if all key is translated 
                    remaining_keys = set(prompt_keys) - set(response[res_keys].keys())
    
                    if remaining_keys:
                        """ If all keys are not translated then new prompt is generated for restranslation """
                        new_prompt = reamaining_prompt(prompt=user_prompt,
                                                    response=response,
                                                    remaining_keys= remaining_keys)
                        
                        remaining_response = translate_chatgpt ( system_prompt=user_prompt, user_prompt=new_prompt, client=open_api_client,)
                        response = {**response, **remaining_response}
                        logger_ocr.debug(f"----------- remaining keys {remaining_keys} \n new response : {response}")
            except Exception as e:
                save_error(f"Error at translate_image_and_draw : {e}")
                logger_ocr.error("Error in  translation ", e)
                response= {}
                # else:
                #     response = {}
            
            # it uses manga pagae analyazer object to draw 
            try:
                values["object"].translate_and_draw(response, translated_path)
            except Exception as e:
                save_error(f"Error at translate_image_and_draw drawing image : {e}")
                logger_ocr.error("Error in  drawing image ", e)
            
def upload_in_parallel(page_number, languages,bucket,source_file_path,epissode_id,urls):
    """
    Upload image to google cloud bucket
    """

    # for page_number in index:    
    logger_ocr.info(f" Translalted page number {page_number}")
    # print(page_number,"  ----- Translated page number ")
    page_info = {"index": int(page_number), "images": []}
    page_number = str(page_number)+".jpg"
    for language in languages:
        image_path = os.path.join(source_file_path, language, page_number)
        public_url = upload_image(bucket,image_path, f"{epissode_id}/{language}/{page_number}")
        page_info["images"].append({"language": language, "url": public_url})
    urls.append(page_info)

@celery.task
def upload_to_gcs(bucket_name, source_file_path, destination_blob_folder, credentials_file,epissode_id,index,single=False):
    """
    Celery task to upload files to Google Cloud Storage (GCS) and generate public URLs.

    This function handles the upload of translated manga images to a specified Google Cloud Storage bucket. 
    returns a list of public URLs for the uploaded images. The upload process is parallelized using a thread pool executor for efficiency.

    Parameters:
        - bucket_name (str): The name of the GCS bucket where the files will be uploaded.
        - source_file_path (str): The local directory path containing the translated image files to be uploaded.
        - destination_blob_folder (str): The target folder path within the GCS bucket where the files will be stored.
        - credentials_file (str): Path to the Google Cloud service account JSON credentials file.
        - epissode_id (str): The unique identifier for the manga episode, used in the upload process.
        - index (list): A list of page indices that correspond to the images to be uploaded.
        - single (bool): A flag indicating whether to upload a single file or multiple language versions. Defaults to `False`.

    Workflow:
        1. Initializes the GCS client using the provided credentials file.
        2. Retrieves the target GCS bucket based on `bucket_name`.
        3. If `single` is `True`, uploads a single file to the specified GCS location and makes it public, returning the public URL.
        4. If `single` is `False`, lists all languages in the source directory and uploads each page in parallel for all languages.
        5. Submits the `upload_in_parallel` function to the thread pool executor for each page in `index`, handling the uploads concurrently.
        6. Collects and logs the public URLs for all uploaded images.
        7. Returns the list of public URLs.

    Returns:
        - urls (list): A list of dictionaries containing the page index and the corresponding public URL for each uploaded image.
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
    # print(os.listdir(os.path.join(source_file_path, languages[0])),"Paths    --------")
    future_results = [executor.submit(upload_in_parallel, page_number,languages,bucket,source_file_path,epissode_id,urls ) for page_number in index]

    results = [future.result() for future in future_results]

    logger_ocr.info(f" Uploaded urls: {urls}")
    return urls

@celery.task
def generate_story_parallel(prompts,image_data,extracted_datas):
    """
    Celery task to generate story translations in parallel using a thread pool.

    This function concurrently generates story translations for a list of prompts by submitting tasks to a thread pool executor. 
    The function is designed to process multiple prompts simultaneously, improving the efficiency of the translation process.

    Parameters:
        - prompts (list): A list of prompt data generated from the manga images. Each prompt contains information needed to generate the story translation.
        - image_data (dict): The JSON object containing details about the manga, including title, source language, and other metadata.
        - extracted_datas (list): A list to collect data extracted during the story generation process.

    Workflow:
        1. Submits the `generate_story` function to the thread pool executor for each prompt in the `prompts` list.
        2. Collects the results of the story generation tasks after they are completed.
        3. Returns the list of results from the story generation tasks.

    Returns:
        - results (list): A list of results from the `generate_story` function for each prompt, representing the generated story translations.
    """

    future_results = [executor.submit(generate_story, prompt,image_data,extracted_datas ) for prompt in prompts]

    results = [future.result() for future in future_results]
    return results

@celery.task
def translate_and_draw_parallel(extracted_datas,target_language, translated_path,image_data,specific_intructions_prompt):
    """
    Celery task to translate and draw text on manga images in parallel using a thread pool.

    This function handles the translation of text and the drawing of translated text onto manga images in parallel. 
    It submits tasks to a thread pool executor, where each task processes one image from the `extracted_datas` list. 
    This parallel approach speeds up the translation and drawing process.

    Parameters:
        - extracted_datas (list): A list of dictionaries containing the extracted data for each image, including the original prompt, story data, and related metadata.
        - target_language (list): A list of target languages (e.g., ['en'] for English) for the translation.
        - translated_path (str): The directory path where the translated images will be saved.
        - image_data (dict): The JSON object containing details about the manga, including series title, chapter ID, and other metadata.
        - specific_intructions_prompt (str): A string containing any custom translation instructions that should be applied.

    Workflow:
        1. Submits the `translate_image_and_draw` function to the thread pool executor for each `extracted_data` item in the `extracted_datas` list.
        2. Waits for all translation and drawing tasks to complete.
        3. Each task translates the extracted text and draws the translated text onto the corresponding image, saving the output in the specified `translated_path`.

    Note:
        - This function does not return any value; it performs the translations and saves the images to the specified directory.
    """


    future_results = [executor.submit(translate_image_and_draw, extracted_data,target_language, translated_path,image_data,specific_intructions_prompt ) for extracted_data in extracted_datas]

    [future.result() for future in future_results]



async def after_completion(response,webhook_url):
    """
    This function sends a POST request to a webhook URL of web app.
    The function uses an authorization token to securely transmit the response data containing the translated image information.

    Parameters:
        - response (dict): The JSON object containing the result of the translation process, including the manga ID, source language, and URLs of the translated pages.
        - webhook_url (str): The URL of the webhook.
    """
    headers = {
        'Authorization': "Bearer "+cfg.COMITEE_WEBHOOK_TOKEN 
    }
    async with httpx.AsyncClient() as client:
        try :
            logger_ocr.info(f" Webhook ulr ===  {webhook_url}")
            response = await client.post(webhook_url+"/webhooks", json= response,headers=headers)
            logger_ocr.info(f" Comitee webhook  {response}")
        except Exception as e:
            save_error(f"Error at after_completion:-  {e}")
            logger_ocr.error(f" Error in comite wep app webhook: with exceptions {e} ------------ ") 
    

def process_image_in_batches(path, batch_size):
    """
    Generator function to process images in batches.

    This function takes a list of image paths and divides them into smaller batches of a specified size. 
    It yields each batch sequentially, allowing for efficient processing of large datasets in manageable chunks.

    Parameters:
        - path (list): A list of image file paths to be processed.
        - batch_size (int): The number of images to include in each batch.

    Workflow:
        1. Iterates through the list of image paths in steps of `batch_size`.
        2. Yields a sublist (batch) of image paths containing up to `batch_size` elements.

    Yields:
        - list: A sublist of image paths representing a single batch of images.
    """

    for i in range(0,len(path), batch_size):
        yield path[i:i+batch_size]


def cleanup_memory():
    gc.collect()
    time.sleep(1)

async def translate_img(image_data,webhook_url):
    """
    Asynchronous function to handle the complete translation process of manga images.

    This function performs the entire workflow for translating manga images, including downloading images, processing them in batches, generating prompts, translating text, drawing translated text on images, and uploading the final translated images to a Google Cloud Storage (GCS) bucket. The function also handles cleanup and notifies a webhook upon completion.

    Parameters:
        - image_data (dict): The JSON object containing manga details, including:
            - "id" (str): Unique identifier for the manga.
            - "title" (str): Title of the manga chapter or episode.
            - "series_title" (str): Title of the manga series.
            - "source_language" (str): Source language code (e.g., 'ja' for Japanese).
            - "target_language" (list): List of target languages (e.g., ['en'] for English).
            - "translation_dictionary" (dict): Custom translation mappings for specific phrases.
            - "pages" (list): List of page information with `index` and `image_url`.
        - webhook_url (str): The webhook URL to send the final processed results.
    Returns:
        - response (dict): A JSON object containing the following:
            - "id" (str): Unique identifier for the manga.
            - "source_language" (str): Source language of the manga.
            - "pages" (list): URLs of the translated pages uploaded to the GCS bucket.
    """

    global open_api_client
    if open_api_client == None:
        load_resources()

    # create root folder according to the id of the epissode  
    root_folder = os.path.join(cfg.IMAGE_FOLDER,str(image_data["id"]))
    logger_ocr.info(f"{root_folder}---------------------------- ")
    os.makedirs(root_folder, exist_ok=True)
    
    target_language= image_data["target_language"]
    logger_ocr.info(target_language)
    save_error(f"---- New manga Start ----  \n Manga : {image_data['title']} id: {image_data['id']}")
    

    chapter_folder = os.path.join(root_folder, str(image_data["id"]))
    os.makedirs(chapter_folder, exist_ok=True)

    # path to store translated image
    translated_path = os.path.join(chapter_folder,"translated_img") 

    # blob path to store translated image in the google bucket   
    destination_blob = os.path.join(cfg.BUCKET_FOLDER,os.path.join(image_data["series_title"],str(image_data["id"])))

    """ Download image in multiprocessing """    
    download_image_loop = asyncio.get_event_loop()
    await download_image_loop.run_in_executor(None, lambda:download_image_parallel( image_data["pages"], chapter_folder))

    # geting index of the image 
    index = [ img_index["index"] for img_index in image_data["pages"]]
    image_paths = [str(p) for p in Path(chapter_folder).glob("*.jpg")] 

    # Retriving only images from the directory  as give on index
    image_paths = [page for page in image_paths if int( page.split(".jpg")[0].split("/")[-1]) in index]

    # sorting image 
    sorted_path = natsorted(image_paths) 

    # Retriving transalation dictionary if it is null setting specific_intructions_prompt to empty
    try:
        translation_dictionary = image_data["translation_dictionaries"]
        if translation_dictionary:
            specific_intructions_prompt = "\nSpecific Custom Instructions:\n"+ specific_instruction(translation_dictionary=translation_dictionary)
        else:
            specific_intructions_prompt=""
    except Exception as e:
        save_error(f"Error ar specific_intructions_prompt {e}")
        specific_intructions_prompt=""
    
    for batch_path in process_image_in_batches(sorted_path,cfg.BATCH_SIZE):
        """
        Processing image transaltion on batches to handel memory insufficient issue.
        """
        print("Image paths ------------------------- ",batch_path)


        # Extracting context of the prompt
        prompts = generate_prompt(image_paths=batch_path, title=image_data["title"])

    
        [os.makedirs(os.path.join(translated_path,translated_language), exist_ok=True) for translated_language in image_data["target_language"]]
        translated_path = os.path.join(chapter_folder,"translated_img")
        os.makedirs(translated_path, exist_ok=True)
        # print("Prompts ::::: *** \n ",prompts, " \n *******************")

        extracted_datas= []

        datas = generate_story_parallel(prompts,image_data,extracted_datas) 
        
        logger_ocr.info  ("*********** All data stored on vector database ***************")
        translate_and_draw_parallel(extracted_datas,target_language, translated_path,image_data,specific_intructions_prompt)

       

        cleanup_memory()

        #  """
        
    # Translated image gets upload to the google cloud bucket
    urls = upload_to_gcs(cfg.BUCKET_NAME,translated_path,destination_blob, cfg.CREDENTIALS_FILE,image_data["id"],index)
    # urls = upload_to_gcs(cfg.BUCKET_NAME,translated_path,destination_blob, cfg.CREDENTIALS_FILE,image_data["id"],index)
    # urls= {}
    
    response = {
        
        "id": image_data["id"],
        "source_language": image_data["source_language"],
        "pages": urls
       
        }
    
    # the uploaded url is send to the web app
    await after_completion(response,webhook_url)

    # delete the root folder after the translation process is complete
    delete_folder(root_folder)
    return response

@signals.task_failure.connect
def handle_task_failure(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **kw):
    logger_ocr.error(f'Task {sender.name} [{task_id}] raised exception: {exception}')
    save_error(f'Task {sender.name} [{task_id}] raised exception: {exception}')

@celery.task(bind=True)
def translate_process(self,image_data,webhook_url):
    """
    Celery task for processing manga image translation.
    This function handles the translation of manga images as an asynchronous Celery task. 
    The task is bound to the Celery instance (`self`), allowing access to task-related metadata and methods.
    Note: self parameter is required slef is the Celery task instance, providing context for task execution.
    """
    loop = asyncio.get_event_loop()
    logger_ocr.info(image_data)
    result = loop.run_until_complete(translate_img(image_data,webhook_url))
    return result

if __name__ == "__main__":
    import asyncio
    image_data= {
    "id": "1",
    "title": "Spy",
    "series_title": "Spy x family",
    "source_language": "ja",
    "target_language": [
        "en",
        "zh"
    ],
    "pages": [
        {
            "index": 0,
            "image_url": "https://cdn.kumacdn.club/wp-content/uploads/S/Spy%20X%20Family/Chapter%2099/26.jpg"
        },
        {
            "index": 27,
            "image_url": "https://cdn.kumacdn.club/wp-content/uploads/S/Spy%20X%20Family/Chapter%2099/27.jpg"
        },
        {
            "index": 28,
            "image_url": "https://cdn.kumacdn.club/wp-content/uploads/S/Spy%20X%20Family/Chapter%2099/28.jpg"
        }
    ]
}   
    webhook_url ="https://www.staging.com"
    print( asyncio.run(translate_img(image_data=image_data,webhook_url= webhook_url)))

    