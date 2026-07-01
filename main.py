import io
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("burns_models.h5")

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

CLASS_NAMES = {
    0: "Degree 1",
    1: "Degree 2",
    2: "Degree 3"
}


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))

    img.save("temp_image.jpg")

    img = img.convert("RGB")
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img = np.asarray(img, dtype=np.float32)

    img = np.expand_dims(img, axis=0)

    return img


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    image_bytes = await image.read()

    image_tensor = preprocess_image(image_bytes)

    predictions = model.predict(image_tensor, verbose=0)[0]

    class_index = int(np.argmax(predictions))

    confidence = float(predictions[class_index])
    print("executed")
    print(predictions)
    print(confidence)

    return {
        "classification": class_index + 1,
        "confidence": round(confidence * 100, 2)
    }