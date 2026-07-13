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

model = tf.keras.models.load_model("burns_model_final_without_softmax_b8e30lr00001.keras")

ODIN_THRESHOLD = 0.3422737121582031

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

CLASS_NAMES = {
    0: "Degree 1",
    1: "Degree 2",
    2: "Degree 3"
}


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))

    img = img.convert("RGB")
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img = np.asarray(img, dtype=np.float32)

    img = np.expand_dims(img, axis=0)

    return img


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    image_bytes = await image.read()

    image_tensor = preprocess_image(image_bytes)

    odin_confidence = odin_score(model, image_tensor, T=100, epsilon=0.3)

    print("ODIN Confidence:", odin_confidence)

    if(odin_confidence < ODIN_THRESHOLD):
        return {
            "classification": -1,
            "confidence": round(odin_confidence * 100, 2)
        }

    predictions = model.predict(image_tensor, verbose=0)[0]

    probs = tf.nn.softmax(predictions).numpy()

    class_index = int(np.argmax(probs))

    confidence = float(probs[class_index])
    print("executed")
    print(probs)
    print(confidence)

    return {
        "classification": class_index + 1,
        "confidence": round(confidence * 100, 2)
    }

def odin_score(model, image, T, epsilon):

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)

        logits = model(image, training=False)
        scaled_logits = logits / T

        pred = tf.argmax(scaled_logits, axis=1)

        loss = tf.keras.losses.sparse_categorical_crossentropy(
            pred,
            scaled_logits,
            from_logits=True
        )

    grad = tape.gradient(loss, image)

    perturbed = image - epsilon * tf.sign(grad)

    perturbed = tf.clip_by_value(
    perturbed,
    0.0,
    255.0
    )

    logits = model(perturbed, training=False)
    scaled_logits = logits / T

    probs = tf.nn.softmax(scaled_logits)

    return float(tf.reduce_max(probs))