import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

def predict_single_image(image, model):
    try:
        # Ensure the image is a PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Resize image to 150x150
        image = image.resize((150, 150))

        # Convert the image to a numpy array
        img_array = img_to_array(image)

        # Rescale the image
        img_array = img_array / 255.0

        # Add an additional dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Make the prediction
        prediction = model.predict(img_array)

        # Interpret the prediction
        if prediction[0][0] > 0.5:
            return "Normal", float(prediction[0][0])
        else:
            return "Acne", float(1 - prediction[0][0])
    except Exception as e:
        print(f"Error in predict_single_image: {str(e)}")
        return "Error", 0.0

# You can add a main block for testing if needed
if __name__ == "__main__":
    print("Model loaded and ready for predictions.")
    # Add any test code here
