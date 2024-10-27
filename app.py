# Flask and CORS for API setup
from flask import Flask, request, redirect
from flask_cors import CORS

# Standard library imports
import os
import io
import base64
import json
from collections import defaultdict
from io import BytesIO

# Logging (if required later)
from logger import setup_logger  # Ensure this is used somewhere in the code or remove it
# Environment variables
from dotenv import load_dotenv

# Machine learning and image processing libraries
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Flask and CORS for API setup
from flask import Flask, request, redirect
from flask_cors import CORS

# OpenAI integration
import openai

# Custom modules for face detection and classification
import detection as detect
import acne_agent
import index
import config
import EX_AI

# Classifier imports
from classifier.ViT import ViTWithDropout, load_entire_vit_model
from classifier.resNext import CustomResNeXt, load_resNext_model
from classifier.rf import load_rf_with_feature_extractor
from classifier.service import classify_acne_image
from logger import setup_logger
logger = setup_logger()
# Asynchronous processing library
import asyncio

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model_name = os.getenv("OPENAI_MODEL_NAME")



# new ======
import detection as detect
import acne_agent
from flask import Flask, request, redirect
from flask_cors import CORS
import base64
import index
from PIL import Image
import openai
from io import BytesIO
import io
import EX_AI
import ViT_xai
from dotenv import load_dotenv
import os

load_dotenv()

import random
import json



def load_treatments_from_json(filename='treatment/self_treatments.json'):
    with open(filename, 'r') as file:
        treatments = json.load(file)
    return treatments


def select_treatment(treatments):
    return random.choice(list(treatments.values()))


def load_recommendation_from_json(filename='treatment/recommendation.json'):
    with open(filename, 'r') as file:
        recommendation = json.load(file)
    return recommendation


def select_recommendation(recommendation):
    return random.choice(list(recommendation.values()))


api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME")

quick_p_scan_model = ViT_xai.get_vit_model_architecture(final_layer_neurons=6, include_extra_layer=True)
quick_p_scan_model.load_weights("model/ViT_acne.h5")

Vit_base_model = ViT_xai.vit_XAI_create_model()
Vit_base_model.load_weights("model/ViT_acne.h5")
vit_b16_layer = Vit_base_model.layers[0]

acne_classes = ['Cyst',
                'Pustules',
                'black & white heads',
                'normal',
                'papules and nodules',
                'scars']

openai.api_key = (api_key)
app = Flask(__name__)
cors = CORS(app)


def stringToImage(base64_string):
    format, img_str = base64_string.split(';base64,')
    ext = format.split('/')[-1]
    imgdata = base64.b64decode(img_str)

    return Image.open(io.BytesIO(imgdata))


@app.route('/api/upload', methods=['POST'])
def upload_file():
    global predicted_class_name, confidance_score, Condition, message, color

    if request.method == 'POST':
        if 'data' not in request.json:
            print('No file part')
            return redirect(request.url)

        data = request.json['data']

        img = data['Image']

        image = index.stringToImage(img)

        folder_path = "img/"
        new_filename = "new_image_name.png"
        index.save_image_to_folder(image, folder_path, new_filename)
        print("Loading", image)
        image_path = 'img/new_image_name.png'

        total_detections, Oveall_class_confidences, detect_plt_image = detect.detect_and_check(image_path)

        image_with_masks, percentage_acne_area = detect.Segment_and_get_area(image_path)
        buffered = BytesIO()
        image_with_masks.save(buffered, format="PNG")
        img_Mask = base64.b64encode(buffered.getvalue()).decode("utf-8")

        summary_stats, combined_severity, color, description = detect.count_base_severity(Oveall_class_confidences)

        yolo_detection = summary_stats
        acne_severity = combined_severity
        acne_description = description
        all_acne = total_detections

        acne_analysis = acne_agent.analyze_acne_with_openai(api_key, model_name, yolo_detection, acne_severity,
                                                            acne_description, all_acne)

        Clinical_diagnosis, AI_explanation = acne_agent.extract_info(acne_analysis)

        treatments = load_treatments_from_json()
        treatment_list = select_treatment(treatments)
        recommendation = load_recommendation_from_json()
        recommendation_list = select_recommendation(recommendation)

        result = {
            'Color': color,
            'Most Probable Diagnosis': Clinical_diagnosis,
            "Overall Explanation": AI_explanation,
            "Recommendations": recommendation_list,
            "Self-Treatment Options": treatment_list,
            "Diagnos Set": summary_stats,
            "Total Detection": total_detections,
            "Total Area": percentage_acne_area,
            'severity': combined_severity,
            "pltImage": detect_plt_image,
            "maskImage": img_Mask
        }
        print(result)
        return {"result": result}


@app.route('/api/xai', methods=['POST'])
def EXAI():
    if request.method == 'POST':
        if 'data' not in request.json:
            print('No file part')
            return redirect(request.url)

        # model_version = request.json['model_version']

        data = request.json['data']

        img = data['Image']

        image = index.stringToImage(img)

        folder_path = "img/"
        new_filename = "eai.png"
        index.save_image_to_folder(image, folder_path, new_filename)
        print("Loading", image)
        image_path = 'img/eai.png'

        img1,predicted_classes1,second_predicted_class2,third_predicted_class3 = EX_AI.process_and_visualize(image_path)

        predicted_class, second_predicted_class, third_predicted_class = EX_AI.ViT_predict(image_path,Vit_base_model,acne_classes)

        buffered = BytesIO()
        img1.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        vit_image = ViT_xai.vit_print_image_paths(image_path, vit_b16_layer)

        result = {
            "First_Predicted_Class": predicted_class,
            "second_predicted_class": second_predicted_class,
            "third_predicted_class": third_predicted_class,
            "image": vit_image,
            "img_str": img_str

        }
        return {"resultex": result}


@app.route('/api/classification', methods=['POST'])
def classification():
    if request.method == 'POST':
        if 'data' not in request.json:
            print('No file part')
            return redirect(request.url)

        # model_version = request.json['model_version']

        data = request.json['data']

        img = data['Image']

        image = index.stringToImage(img)

        folder_path = "img/"
        new_filename = "eai.png"
        index.save_image_to_folder(image, folder_path, new_filename)
        print("Loading", image)
        image_path = 'img/eai.png'

        # img, predicted_class, second_predicted_class, third_predicted_class = EX_AI.process_and_visualize(image_path)
        # buffered = BytesIO()
        # img.save(buffered, format="PNG")
        # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        vit_image = ViT_xai.vit_print_image_paths(image_path, vit_b16_layer)

        result = {
            "First_Predicted_Class": "predicted_class",
            "second_predicted_class": "second_predicted_class",
            "third_predicted_class": "third_predicted_class",
            "image": vit_image,
            "img_str": None

        }
        return {"resulClassification": result}


binary_classifier_model = tf.keras.models.load_model('model/binary_acne_classifier.h5', compile=False)
binary_classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rf_feature_extractor, rf_model = load_rf_with_feature_extractor()
vit_model = load_entire_vit_model()
resNext_model = load_resNext_model()

@app.route('/api/acne/classifier', methods=['POST'])
def upload_image():
    # Call the async function and run it synchronously using asyncio.run()
    result = asyncio.run(classify_acne_image(
        file=request.files['file'],
        binary_classifier_model=binary_classifier_model,
        rf_feature_extractor=rf_feature_extractor,
        rf_model=rf_model,
        vit_model=vit_model,
        resNext_model=resNext_model,
        logger=logger
    ))

    return result

if __name__ == "__main__":
    app.run(debug=False, threaded=False)  # Disable multi-threading
