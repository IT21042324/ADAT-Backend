import cv2  # OpenCV for image processing
import torch
from mtcnn import MTCNN  # MTCNN for face detection
import requests
import json
import os

# Initialize the MTCNN face detector once, so it can be reused
detector = MTCNN()

def throwErrorIfNoFacesAreAvailableInImage(faces):
    print(f'faces: {faces}')
    # Check if no faces are detected
    if not faces:
        raise ValueError("No face detected in the image. Please provide an image with a visible face.")

def check_image_quality(face_image):
    """
    Function to check the quality of the face image using SightEngine API.

    Parameters:
    face_image (np.array): Cropped face image in RGB format.

    Returns:
    tuple: Quality score of the face image and an indication if it's sufficient.

    Raises:
    ValueError: If there is an error with the API request.
    """
    # Save the image temporarily for quality check
    temp_filename = 'temp_face.jpg'
    cv2.imwrite(temp_filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

    # Set the parameters and file for the API request
    params = {
        'models': 'quality',
        'api_user': os.getenv('SIGHTENGINE_API_USER'),
        'api_secret': os.getenv('SIGHTENGINE_API_SECRET')
    }
    files = {'media': open(temp_filename, 'rb')}

    try:
        # Make the request to the API
        r = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
        output = json.loads(r.text)
    finally:
        # Ensure that the file is closed and deleted
        files['media'].close()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # Get the quality score from the response
    quality_score = output.get('quality', {}).get('score', 0.0)

    # Determine if the quality is sufficient
    is_quality_sufficient = quality_score > 0.4

    # Return both the quality score and the indication if it's sufficient
    return quality_score, is_quality_sufficient

def extract_face(img):
    """
    Function to detect and extract the largest face region from an image using MTCNN Face Detection.

    Parameters:
    img (np.array): Input image in RGB format (as expected by MTCNN).

    Returns:
    dict: Dictionary containing the cropped face region, quality score, and an indication of quality.

    Raises:
    ValueError: If no faces are detected in the image.
    """
    # Detect faces in the input image using MTCNN
    faces = detector.detect_faces(img)

    # Check if no faces are detected
    if not faces:
        raise ValueError("No face detected in the image. Please provide an image with a visible face.")

    # If multiple faces are detected, select the largest one based on the bounding box area
    largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])  # box[2] * box[3] is width * height

    # Get the bounding box of the largest detected face
    x, y, width, height = largest_face['box']

    # Apply margin if needed
    margin = 10
    x_min = max(0, x - margin)
    y_min = max(0, y - margin)
    x_max = min(img.shape[1], x + width + margin)
    y_max = min(img.shape[0], y + height + margin)

    # Crop the face region from the original image
    face_image = img[y_min:y_max, x_min:x_max]

    # Check the image quality of the extracted face
    quality_score, is_quality_sufficient = check_image_quality(face_image)

    # Return the cropped face region and the quality information
    return {
        "face_image": face_image,
        "quality_score": quality_score,
        "is_quality_sufficient": is_quality_sufficient
    }
