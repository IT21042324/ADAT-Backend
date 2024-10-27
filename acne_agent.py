from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import openai
import json
import os


def analyze_acne_with_openai(api_key, model_name, yolo_detection, acne_severity, acne_description, total_detections):
    openai.api_key = api_key

    acne_model_output = {
        "yolo_detection": yolo_detection,
        "acne_severity": acne_severity,
        "acne_description": acne_description,
        "total_detections": total_detections
    }

    system_prompt = """You are an expert dermatologist with extensive knowledge in AI-assisted dermatological diagnosis. Your task is to analyze the output from an advanced acne detection AI model and provide a comprehensive assessment. Use your expertise to interpret the data and offer valuable insights to the patient.
    Parse the provided output to extract information about:
        1. YOLO Detection Results (including detection counts, average confidence, and max confidence for different acne types)
        2. Acne Severity
        3. Acne Description
        4. Total Acne cases detected

        Based on this information, please strictly provide a detailed analysis including the following only:
        
        1. Clinical Explanation:
           Need to give a clinical explanation Based on the AI detection results and your expert knowledge, what is the most likely diagnosis for this patient's skin condition? Provide a concise yet informative explanation.
        
        2. AI Explanation:
            Interpret the AI model's outputs as a paragraph, including the detection counts, confidence levels, and severity assessment. Explain how these factors contribute to your diagnosis and what they mean for the patient's skin health.
        
        Remember to maintain a professional and empathetic tone throughout your analysis. Your goal is to provide clear, actionable advice based on the AI model's output and your dermatological expertise. please don't add the *'s for the final output """

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"Analyze the following acne model output:\n\n{json.dumps(acne_model_output, indent=2)}"}
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Extract and return the generated analysis
    return response.choices[0].message["content"]


def extract_info(acne_analysis):
    sections = acne_analysis.strip().split('\n\n')
    # Remove the numbering from each section header and text
    Clinical_diagnosis = sections[0].split(':', 1)[1].strip()
    AI_explanation = sections[1].split(':', 1)[1].strip()
    # overall_explanation = sections[2].split(':', 1)[1].strip()
    # # Extract lists for the last two sections
    # recommendations = [line.strip('- ').strip() for line in sections[3].split('\n')[1:] if line.strip()]
    # self_treatment_options = [line.strip('- ').strip() for line in sections[4].split('\n')[1:] if line.strip()]
    return Clinical_diagnosis, AI_explanation
