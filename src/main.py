import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from typing import List

import numpy as np
import redis
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from config import REDIS_HOST, REDIS_PORT

model = tf.keras.models.load_model("model1", compile=False)

labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

app = FastAPI()

redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)

@app.get('/help/') 
def help_func():
    return f"Данный сервис выполняет классификацию изображений на базе датасета Cifar10. " \
            f"Изображения можно загружать в формате нескольких файлов или zip-архива, на выходе вы получите json-файлы с разметкой. " \
            f"Допустимые форматы изображения: .jpg, .jpeg и .png. " \
            f"Помимо этого, вы можете узнать о статистике и состоянии работы сервиса. " \
            f"Доступные команды:" \
            f"GET: /help/ - получить справку. " \
            f"GET: /print-labels/ - получить список классов изображений. " \
            f"POST: /classify-images/ - выполнить классификацию одного или нескольких изображений. " \
            f"POST: /classify-zip/ - загрузить архив и выполнить классификацию. "\
            f"GET: /health/ - узнать статус работы сервиса. " \
            f"GET: /stats/ - узнать кол-во запросов, среднее время работы запроса и общее время. "


@app.get("/print-labels/")
def get_class_labels():
    result = redis_client.get("labels_result")
    if result:
        return JSONResponse(content={"labels": result.decode("utf-8").split(",")}, status_code=200)

    redis_client.set("labels_result", ",".join(labels))

    return {"labels": labels}


@app.post("/classify-images/")
def classify_images(files: List[UploadFile] = File(...)):
    predictions = []
    for file in files:
        pred_from_db = redis_client.get(file.filename)
        if pred_from_db:
            predictions.append(json.loads(pred_from_db.decode().replace("'", '"')))
        else:
            contents = file.file.read()

            image = Image.open(io.BytesIO(contents))
            image = image.resize((32, 32))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            class_label = labels[predicted_class]
            predictions.append({"filename": file.filename, "label": class_label})

            redis_client.set(file.filename, str({"filename": file.filename, "label": class_label}))

    return predictions


def load_zip(archive: UploadFile):
    predictions = []
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, archive.filename)
        with open(file_path, "wb") as file_object:
            shutil.copyfileobj(archive.file, file_object)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        for dirname in os.listdir(temp_dir):
            if os.path.isdir(temp_dir + '/' + dirname):
                for filename in os.listdir(temp_dir + '/' + dirname):
                    if filename.endswith((".jpg", ".jpeg", ".png")):
                        pred_from_db = redis_client.get(filename)
                        if pred_from_db:
                            predictions.append(json.loads(pred_from_db.decode().replace("'", '"')))
                        else:
                            image_path = os.path.join(temp_dir + "/" + archive.filename.split(".")[0], filename)
                            image = Image.open(image_path)
                            image = image.resize((32, 32))
                            image = np.array(image) / 255.0
                            image = np.expand_dims(image, axis=0)

                            prediction = model.predict(image)
                            predicted_class = np.argmax(prediction)
                            class_label = labels[predicted_class]
                            predictions.append({"filename": filename, "label": class_label})

                            redis_client.set(filename, str({"filename": filename, "label": class_label}))

    return predictions


@app.post("/classify-zip/")
def classify_zip(archive: UploadFile = File(...)):
    labels = load_zip(archive)

    return labels



@app.get('/health/')
def check_func():
    try:
        return "OK"
    except:
        return HTTPException(status_code=500, detail='Наблюдаются проблемы с сервисом')


class Stats:
    requests_received = 0
    processing_time = 0


@app.middleware("http")
def stats_middleware(request, call_next):
    start_time = time.time()
    response = call_next(request)
    Stats.requests_received += 1
    processing_time = time.time() - start_time
    Stats.processing_time += processing_time
    return response


@app.get("/stats/")
def usage_stats():
    return {
        "requests_received": Stats.requests_received,
        "average_response_time": Stats.processing_time / Stats.requests_received if Stats.requests_received > 0 else 0,
        "total_processing_time": Stats.processing_time,
    }
