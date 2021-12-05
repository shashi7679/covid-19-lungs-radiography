from fastapi import FastAPI, File, UploadFile
from tensorflow.keras import models
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
import requests

model = models.load_model('covid-19-lungs.h5')
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']


def cvt2image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image


app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello I am alive!!!!"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes = await file.read()
    image = cvt2image(bytes)
    image = image / 255.0
    image = cv2.resize(image, (60, 60))
    image = np.array(image).reshape(1, 60, 60, 1)
    #image = np.expand_dims(image, 0)
    predict = model.predict(image)
    prediction_class = class_names[np.argmax(predict[0])]
    confidence = np.max(predict[0])
    return {'class': prediction_class, 'confidence': float(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
