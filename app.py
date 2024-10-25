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

# Environment variables
from dotenv import load_dotenv

# Logging (not used in this code)
from logger import setup_logger

# Machine learning and image processing libraries
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO
import matplotlib.pyplot as plt

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
from classifier.acne_classifier import weighted_vote_single_image
from classifier.classifier import predict_single_image
from classifier.face_extractor import extract_face
from classifier.resNext import CustomResNeXt, load_resNext_model
from classifier.rf import load_rf_with_feature_extractor
from classifier.service import classify_acne_image

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model_name = os.getenv("OPENAI_MODEL_NAME")

openai.api_key = (openai_api_key)
app = Flask(__name__)
cors = CORS(app)

logger = setup_logger()

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

        # model_version = request.json['model_version']

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

        summary_stats, combined_severity, color, description = detect.count_base_severity(Oveall_class_confidences,
                                                                                          percentage_acne_area)

        yolo_detection = summary_stats
        acne_severity = combined_severity
        acne_description = description
        all_acne = total_detections

        acne_analysis = acne_agent.analyze_acne_with_openai(openai_api_key, openai_model_name, yolo_detection, acne_severity,
                                                            acne_description, all_acne)

        Clinical_diagnosis, AI_explanation = acne_agent.extract_info(acne_analysis)

        result = {
            'Color': color,
            'Most Probable Diagnosis': Clinical_diagnosis,
            "Overall Explanation": AI_explanation,
            "Recommendations": config.Reccomentdation,
            "Self-Treatment Options": config.selfTreatments,
            "Diagnos Set": summary_stats,
            "Total Detection": total_detections,
            "Total Area": percentage_acne_area,
            'severity': combined_severity,
            "pltImage": None,
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

        img, predicted_class, second_predicted_class, third_predicted_class = EX_AI.process_and_visualize(image_path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        result = {
            "First_Predicted_Class": predicted_class,
            "second_predicted_class": second_predicted_class,
            "third_predicted_class": third_predicted_class,
            "image": img_str
        }
        return {"resultex": result}


@app.route('/api/classificationforxai', methods=['POST'])
def classificationforxai():
    if request.method == 'POST':
        if 'data' not in request.json:
            print('No file part')
            return redirect(request.url)

        # model_version = request.json['model_version']

        data = request.json['data']

        img = data['Image']

        image = index.stringToImage(img)

        folder_path = "img/"
        new_filename = "classification.png"
        index.save_image_to_folder(image, folder_path, new_filename)
        print("Loading", image)
        image_path = 'img/classification.png'

        img, predicted_class, score, second_predicted_class, second_score = EX_AI.process_and_visualize(image_path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        result = {
            "predicted_class": predicted_class,
            "second_predicted_class": second_predicted_class,
            "score": score,
            'second_score': second_score,
            "image": img_str,
            "message": "xxxx"

        }
        return {"resultClassification": result}





binary_classifier_model = tf.keras.models.load_model('model/binary_acne_classifier.h5', compile=False)
binary_classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rf_feature_extractor, rf_model = load_rf_with_feature_extractor()
vit_model = load_entire_vit_model()
resNext_model = load_resNext_model()

@app.route('/api/classification', methods=['POST'])
def classification():
    # Call the function from classification_service.py
    return classify_acne_image(
        file=request.files['file'],
        binary_classifier_model=binary_classifier_model,
        rf_feature_extractor=rf_feature_extractor,
        rf_model=rf_model,
        vit_model=vit_model,
        resNext_model=resNext_model,
        logger=logger
    )

if __name__ == "__main__":
    app.run(debug=False)
