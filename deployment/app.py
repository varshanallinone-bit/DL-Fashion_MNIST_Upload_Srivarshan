from pathlib import Path
import io

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fashion_cnn_model.keras"

app = FastAPI(
    title="Fashion-MNIST CNN Classifier",
    version="1.0.0",
)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Load the model once when the service starts.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

app.mount(
    "/css",
    StaticFiles(directory=str(BASE_DIR / "css")),
    name="css",
)

app.mount(
    "/js",
    StaticFiles(directory=str(BASE_DIR / "js")),
    name="js",
)


@app.get("/")
def home():
    return FileResponse(str(BASE_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Fashion-MNIST CNN",
        "classes": len(class_names),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 10 MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("L")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # Match Fashion-MNIST training input: 28x28 grayscale, normalized to [0, 1].
    image = image.resize((28, 28))
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    # CNN input shape: (batch, height, width, channels)
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    logits = model.predict(image_array, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()

    top_indices = np.argsort(probabilities)[::-1][:3]

    predictions = [
        {
            "class": class_names[int(i)],
            "confidence": round(float(probabilities[i]) * 100, 2),
        }
        for i in top_indices
    ]

    return {
        "success": True,
        "filename": file.filename,
        "model": "Fashion-MNIST CNN",
        "predictions": predictions,
    }
