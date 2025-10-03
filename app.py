import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

# --- Model Loading and Setup ---

# NOTE: Since the trained model file 'digit_model.keras' is not included 
# in this single-file environment, the code attempts to load it from the 
# local directory. If the file is not found, a new untrained model 
# architecture is created for demonstration.

MODEL_PATH = "digit_model.keras"

def load_or_create_model():
    """Loads the model or creates the architecture if the file is missing."""
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Pre-trained model loaded successfully from {MODEL_PATH}!")
        else:
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing model architecture with random weights.")
        
        # Define the CNN model architecture exactly as trained in your notebook (Cell 32)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(pool_size=2),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # Build and compile the model to prepare it for prediction
        model.build(input_shape=(None, 28, 28, 1))
        
        print("Model initialized with random weights. Predictions will be random until the correct weights are loaded/trained.")
    
    return model

# Load the model globally
model = load_or_create_model()


def predict_digit(img):
    """
    Pre-processes the input image from Gradio and makes a prediction.
    The pre-processing mirrors the MNIST data preparation.
    """
    
    if img is None:
        return {} 

    # 1. Convert to Grayscale
    # Gradio canvas input is typically RGBA (H, W, 4). 
    # Use cv2.COLOR_RGBA2GRAY for safe conversion.
    img = img.astype(np.uint8) 
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # 2. Resize to 28x28 (Input shape for CNN)
    resized_img = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # 3. Invert Colors, Apply Threshold, and Normalize
    # MNIST is white digit on black background (0=background, 255=digit). 
    # Gradio canvas is often black drawing on white background. We invert it.
    inverted_img = 255 - resized_img
    
    # Use a threshold to sharpen the drawn digit (values below 50 are set to zero)
    _, thresholded_img = cv2.threshold(inverted_img, 50, 255, cv2.THRESH_TOZERO)
    
    # 4. Normalize to [0, 1] and Reshape to (1, 28, 28, 1)
    normalized_img = thresholded_img / 255.0
    
    # Add channel dimension (28, 28) -> (28, 28, 1)
    input_tensor = np.expand_dims(normalized_img, axis=-1)
    
    # Add batch dimension (28, 28, 1) -> (1, 28, 28, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

    # 5. Predict
    predictions = model.predict(input_tensor, verbose=0)
    
    # Format the output for Gradio's Label component (mapping digit string to probability)
    probabilities = predictions[0].tolist()
    return {str(i): probabilities[i] for i in range(10)}

# --- Gradio Interface ---

# Input component: A sketchpad for the user to draw the digit
# The shape is set large for comfortable drawing.
input_canvas = gr.Image(
    shape=(280, 280), 
    image_mode='RGBA', 
    source='canvas', 
    type='numpy', 
    label='Draw a single digit (0-9)',
    tool='sketch',
    # Set a background color for the canvas if needed, or leave it transparent
    # if you want the app's default background to show through.
)

# Output component: A label to show prediction probabilities
output_label = gr.Label(
    num_top_classes=10, 
    label="Prediction Probabilities"
)

# Create the Gradio Interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=input_canvas,
    outputs=output_label,
    title="Handwritten Digit Recognition (MNIST CNN)",
    description=(
        "Draw a digit (0-9) on the canvas above. The Convolutional Neural Network (CNN) "
        "will predict the digit and show the confidence scores below. "
        "Remember to click 'Clear' on the canvas between drawings."
    ),
    examples=[],
    allow_flagging='never',
    live=True
)

if __name__ == "__main__":
    iface.launch()
