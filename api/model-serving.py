from fastapi import FastAPI, File, UploadFile
from tensorflow.keras import models
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

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
    image = np.expand_dims(image, 0)
    image = tf.image.per_image_standardization(image)
    image = np.expand_dims(image, 0)
    image = np.array(image).resize(1, 60, 60, 1)
    image = tf.image.per_image_standardization(image)
    print(image.shape)
    predict = model.predict(image)
    return predict


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
