
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Model and class names
MODEL_PATH = "models/plant_disease_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMAGE_SIZE = (160, 160)

# Load trained model
print("Loading Plant Disease Model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("Model loaded successfully!")


def predict_disease(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image into array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    disease_name = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    print("\n========== PREDICTION RESULT ==========")
    print("Plant Disease:", disease_name)
    print(f"Confidence: {confidence:.2f}%")
    print("=======================================")

    return disease_name, confidence


if __name__ == "__main__":

    image_path = input("Enter leaf image path: ").strip().strip('"')

    predict_disease(image_path)
