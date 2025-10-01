from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ✅ Load model
model = tf.keras.models.load_model("digit_model.keras")

# ✅ Create app
app = FastAPI()

# ✅ CORS (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Preprocess image
def preprocess_image(image: Image.Image):
    image = image.convert("L").resize((28, 28))  # grayscale, 28x28
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.get('/')
def read_root():
    return FileResponse("index.html")

# ✅ Prediction route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))
    return {"predicted_digit": digit}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    





