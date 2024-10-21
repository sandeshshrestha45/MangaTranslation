from python:3.10-slim
EXPOSE 8000
WORKDIR /app
COPY ./app /app/
COPY ./app/requirements.txt ./

RUN apt-get update && apt-get install -y \
    libopencv-dev \  
    libgl1-mesa-dev 
RUN  pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r  requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
