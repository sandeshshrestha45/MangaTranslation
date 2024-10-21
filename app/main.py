import os
import redis
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import HTMLResponse
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from celery.result import AsyncResult
from celery_worker import celery
import config as cfg
from tasks import translate_process
import json

app = FastAPI()
# backend =  redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
redis_server = redis.Redis(host=cfg.REDIS_IP, port=cfg.REDIS_PORT)

@app.get("/")
async def root(request: Request):
    return  {"message": "Api up and working"}


@app.get("/log")
def get_log():
    """
        This returns the log of the application which store in app.log .
    """
    log_file_path = "app.log"
    if not os.path.exists(log_file_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    with open(log_file_path, "r", encoding="utf-8") as file:
        log_content = file.read()

    # Use Pygments to add syntax highlighting
    lexer = get_lexer_by_name("logtalk", stripall=True)
    formatter = HtmlFormatter(linenos='table', full=True, style='colorful')
    highlighted_log = highlight(log_content, lexer, formatter)
    return HTMLResponse(content=highlighted_log)

@app.post("/manga/job-status")
async def jobs(request:Request):
    """
    Handles the POST request to check the status of a manga job using a Celery task ID.

    Args:
        request (Request): The incoming HTTP request containing a JSON payload with the task ID.
        {
            "id": <task_id>
        }

    Returns:
        dict: A dictionary containing the status of the task. Possible values are:
            - "PENDING" if the task is in the queue or hasn't started yet.
            - "SUCCESS", "FAILURE", or other statuses depending on the Celery task's progress.
            - "INVALID_ID" if the provided ID is missing or invalid.
        {"status": "SUCCESS"}
    """
    data = await request.json()
    if data["id"]:
        result = AsyncResult(data["id"], app=celery)
        celery_status = result.status
        if celery_status == "PENDING":
            pending = redis_server.lrange("celery",0,-1)
            pending =[json.loads(data.decode("utf-8")) for data in pending]
            valid_id= False
            for pending_job in pending:
                if data["id"] == pending_job["headers"]["id"]:
                    return {"status":"PENDING"}
            if not valid_id:
                return {"status":"PENDING"}
        else:
            return {"status": celery_status}
    else:
        return {"status":"INVALID_ID"}


    
@app.post("/manga/image-translations")
async def image_translations(request: Request):
    """
    This function is for translating images for testing purpose only this doesnt not run on background.
    It accpet json fromat 

    """
    from tasks import translate_img
    image_data = await request.json()
    webhook_url = cfg.APP_URL
    response = await translate_img(image_data, webhook_url)
    return response



@app.post("/manga/webhook")
async def webhook(request:Request):
    """
    Webhook endpoint for initiating the manga image translation process. It is the main api used buy the web application.

    This asynchronous POST request handler receives manga translation task details in JSON format. The input data includes manga title, 
    series, source and target languages, translation dictionary, and a list of page URLs for translation.

    Parameters:
        - request (Request): The incoming request containing JSON data with manga translation details.
        - The JSON data structure includes:
            - "id" (str): Unique ID of the manga.
            - "title" (str): Title of the manga chapter or episode.
            - "series_title" (str): Title of the manga series.
            - "source_language" (str): Source language code (e.g., 'ja' for Japanese).
            - "target_language" (list): List of target languages (e.g., ['en'] for English).
            - "translation_dictionaries" (dict): Custom translation mappings for specific phrases.
            - "pages" (list): List of page information with `index` and `image_url`.
    Returns:
        JSON response with status code 200 and the task ID.
    """

    image_data = await request.json() 
    webhook_url = cfg.APP_URL
    # start background jobs
    task = translate_process.apply_async(args=[image_data,webhook_url]) 
    cfg.logger_ocr.info(f" Translation process started for {task.id}")
    return {"status":200,"message":"Image transaltion is under process","task_id":task.id}
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)