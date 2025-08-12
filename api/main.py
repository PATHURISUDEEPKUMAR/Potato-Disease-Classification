from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = FastAPI()

Model = tf.keras.models.load_model(r"C:\Users\HP\Desktop\potato_disease\models\1")
Class_names = ["Early_Blight", "Late_Blight", "Healthy"]


@app.get("/hello")
async def hello():
    return "HELLO HW R U "

def read_file_as_img(data) ->np.ndarray:
    return np.array(Image.open(BytesIO(data)))
    
@app.get("/page/{id}/{name}")
async def page(id,name: str):
    return {'data': {
        'name': name,
        'performance': id
    }
    }


@app.post("/predict")
async def predict(file: UploadFile
                  = File(...)):
    image = read_file_as_img(await file.read())
    img_batch = np.expand_dims(image,0)
    prediction = Model.predict(img_batch)
    predicted_class = Class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'Class': predicted_class,
        'confidence': float(confidence)
    }


    