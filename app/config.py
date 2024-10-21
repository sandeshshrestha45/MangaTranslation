import os
import logging
from dotenv import load_dotenv


logging.basicConfig(level=logging.ERROR)
logger_ocr = logging.getLogger("Comitee-ml ")
logger_ocr.setLevel(logging.DEBUG)
fh = logging.FileHandler("app.log")
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger_ocr.addHandler(fh)
logger_ocr.addHandler(ch)

BATCH_SIZE= 20
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(ROOT_PATH,".env")
load_dotenv(dotenv_path)

ERROR_FILE =os.path.join(ROOT_PATH, "error.txt")

IMAGE_FOLDER = os.path.join(ROOT_PATH, "img")
MODEL_PATH = os.path.join(ROOT_PATH,"model/character/new_added/weights/best.pt")
TRANSLATED_IMG = os.path.join(ROOT_PATH,"img/translated_img")
GENDER_MODEL_CONFIG = os.path.join(ROOT_PATH,'model/gender/config.json')
GENDER_MODEL = os.path.join(ROOT_PATH,"model/gender/model.weights.h5")
EBEDDING_MODEL= os.environ.get("EBEDDING_MODEL")

WEAVIET_SCHEMA = os.environ.get("WEAVIET_SCHEMA")

WEAVIET_URL =os.environ.get("WEAVIET_URL")
REDIS_URL= os.environ.get("REDIS_URL")
FASTAPI_URL= os.environ.get("FASTAPI_URL")

REDIS_IP = os.environ.get("REDIS_IP")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_DB = os.environ.get("REDIS_DB")

# GPT_MODEL ="gpt-3.5-turbo-16k"

GPT_MODEL = os.environ.get("GPT_MODEL")
OPEN_AI = os.environ.get("OPEN_AI")
FONT_PATH= os.path.join(ROOT_PATH,os.environ.get("FONT_PATH"))
BUCKET_NAME =os.environ.get("BUCKET_NAME")
CREDENTIALS_FILE = os.path.join(ROOT_PATH,os.environ.get("CREDENTIALS_FILE"))

BUCKET_FOLDER = os.environ.get("BUCKET_FOLDER")

COMITEE_WEBHOOK_TOKEN= os.environ.get("COMITEE_WEBHOOK_TOKEN")
APP_URL= os.environ.get("APP_URL")
# APP_URL ="https://8305-27-34-64-175.ngrok-free.app/"