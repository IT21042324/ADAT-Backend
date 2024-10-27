from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import openai
import json
import index
import os
from dotenv import load_dotenv


def detect_and_check(image_path):
    yolo_model = YOLO("model/best.pt")
    names = yolo_model.names
    '''Overall Detection'''
    Overall_results = yolo_model(image_path, show=False, conf=0.2)
    Oveall_class_confidences = defaultdict(list)
    
    for s in Overall_results:
        for x, confx in zip(s.boxes.cls, s.boxes.conf):
            class_name = names[int(x)]
            Oveall_class_confidences[class_name].append(confx.item())
            
    total_detections = sum(len(confidences) for confidences in Oveall_class_confidences.values())
    print(total_detections)

    for i, r in enumerate(Overall_results):
        im_bgr = r.plot()  
        im_rgb = Image.fromarray(im_bgr[..., ::-1]) 
        plt_image = im_rgb.convert("RGB")

    plt_image = index.image_to_base64(plt_image)
 
    return  total_detections, Oveall_class_confidences,plt_image

def Segment_and_get_area(image_path):
    # Load a pretrained YOLOv8n model
    model = YOLO("model/best.pt")
    
    # Perform inference
    results = model(image_path, conf=0.1)
    
    # Open the original image
    original_image = Image.open(image_path)
    
    # Initialize variables
    alpha_factor = 0.3 
    blue_pixel_count = 0
    total_pixels = original_image.width * original_image.height

    # Apply masks to the image
    for r in results:
        for mask in r.masks.data:
            mask_np = (mask * 255).byte().cpu().numpy()
            rgba_image = Image.new("RGBA", (mask_np.shape[1], mask_np.shape[0]), (0, 0, 255, 0))

            for y in range(mask_np.shape[0]):
                for x in range(mask_np.shape[1]):
                    if mask_np[y, x] > 0:  # This pixel is part of the mask
                        rgba_image.putpixel((x, y), (0, 0, 255, int(mask_np[y, x] * alpha_factor)))
                        blue_pixel_count += 1

            rgba_resized = rgba_image.resize(original_image.size)
            original_image = Image.alpha_composite(original_image.convert("RGBA"), rgba_resized)

    blue_percentage = (blue_pixel_count / total_pixels) * 100
    # original_image = index.image_to_base64(original_image)
    
    return original_image, blue_percentage


def count_base_severity(Oveall_class_confidences):

    # Initialize the summary statistics and the total severity score
    summary_stats = {}
    total_score = 0

    # Define the severity scale for different acne types
    severity_scale = {
        'blackheads': 1,
        'whiteheads': 1,
        'papules': 3,
        'pustules': 2,
        'nodules': 4,
        'dark spots': 2
    }

    # Calculate summary statistics and accumulate total severity score
    for class_name, confidences in Oveall_class_confidences.items():
        average_confidence = sum(confidences) / len(confidences)
        detection_count = len(confidences)
        max_confidence = max(confidences)
        summary_stats[class_name] = {
            'average_confidence': average_confidence,
            'detection_count': detection_count,
            'max_confidence': max_confidence
        }

        if class_name in severity_scale:
            total_score += detection_count * severity_scale[class_name]

    # Calculate severity based on detection count and severity score
    if total_score < 10:
        detection_severity = "Mild"
        color = "Green"
        description = "Acne severity is mild, suggesting few inflammatory lesions and minimal scarring risk."

    elif 10 <= total_score < 25:
        detection_severity = "Moderate"
        color = "Yellow"
        description = "Acne severity is moderate. This condition may cause some discomfort and visible inflammation."

    elif 25 <= total_score < 40:
        detection_severity = "Severe"
        color = "Orange"
        description = "Acne severity is severe, indicating numerous inflamed lesions and higher risk of scarring."
    else:
        detection_severity = "Extremely Severe"
        color = "Red"
        description = "Acne severity is extremely severe. Urgent dermatological consultation is recommended due to " \
                      "significant risk of scarring and severe inflammation. "

    # # Determine severity based on percentage of acne area
    # if percentage_acne_area < 4:
    #     area_severity = "Mild"
    # elif 4 <= percentage_acne_area < 6:
    #     area_severity = "Moderate"
    # elif 6 <= percentage_acne_area < 12:
    #     area_severity = "Severe"
    # else:
    #     area_severity = "Extremely Severe"
    #
    # # Combine both severity assessments to determine final severity
    # severity_levels = ['Mild', 'Moderate', 'Severe', 'Extremely Severe']
    #
    # # Pick the more severe classification between detection-based and area-based severity
    # final_severity = max(severity_levels.index(detection_severity), severity_levels.index(area_severity))
    # combined_severity = severity_levels[final_severity]


    return summary_stats, detection_severity , color, description