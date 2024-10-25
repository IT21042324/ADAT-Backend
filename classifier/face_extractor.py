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


def check_image_quality(face_image, logger):
    """
    Function to check the quality of the face image using SightEngine API.

    Parameters:
    face_image (np.array): Cropped face image in RGB format.
    logger: Logger instance for logging relevant messages.

    Returns:
    tuple: Quality score of the face image and an indication if it's sufficient.

    Raises:
    ValueError: If there is an error with the API request.
    """
    # Save the image temporarily for quality check
    temp_filename = 'temp_face.jpg'
    cv2.imwrite(temp_filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

    # Define two sets of API credentials
    api_credentials = [
        {
            'api_user': os.getenv('SIGHTENGINE_API_USER1'),
            'api_secret': os.getenv('SIGHTENGINE_API_SECRET1')
        },
        {
            'api_user': os.getenv('SIGHTENGINE_API_USER2'),
            'api_secret': os.getenv('SIGHTENGINE_API_SECRET2')
        }
    ]

    # Attempt to check the image quality using both accounts
    for credentials in api_credentials:
        params = {
            'models': 'quality',
            'api_user': credentials['api_user'],
            'api_secret': credentials['api_secret']
        }
        files = {'media': open(temp_filename, 'rb')}

        try:
            # Log the account being used
            logger.info(f"Using SightEngine account: {credentials['api_user']}")

            # Make the request to the API
            r = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
            output = json.loads(r.text)

            # If the request was successful, process the response
            if r.status_code == 200 and 'quality' in output:
                # Get the quality score from the response
                quality_score = output.get('quality', {}).get('score', 0.0)

                # Determine if the quality is sufficient
                is_quality_sufficient = quality_score > 0.4

                logger.info(f"Image quality score: {quality_score}, is sufficient: {is_quality_sufficient}")

                # Ensure the file is closed and deleted
                files['media'].close()
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

                # Return both the quality score and the indication if it's sufficient
                return quality_score, is_quality_sufficient
            else:
                logger.warning(f"Failed to retrieve quality score with account: {credentials['api_user']}")
        except Exception as e:
            logger.error(f"Error with SightEngine account {credentials['api_user']}: {e}")
        finally:
            files['media'].close()

    # If all accounts fail, log and raise an error
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    logger.critical("Both SightEngine accounts failed. Please check the API credentials or service availability.")
    raise ValueError("Both SightEngine accounts failed. Please check the API credentials or service availability.")


def extract_face(img, logger):
    """
    Function to detect and extract the largest face region from an image using MTCNN Face Detection.

    Parameters:
    img (np.array): Input image in RGB format (as expected by MTCNN).
    logger: Logger instance for logging relevant messages.

    Returns:
    dict: Dictionary containing the cropped face region, quality score, and an indication of quality.

    Raises:
    ValueError: If no faces are detected in the image.
    """
    # Detect faces in the input image using MTCNN
    faces = detector.detect_faces(img)

    # Check if no faces are detected
    if not faces:
        logger.error("No face detected in the image.")
        raise ValueError("No face detected in the image. Please provide an image with a visible face.")

    # If multiple faces are detected, select the largest one based on the bounding box area
    largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])  # box[2] * box[3] is width * height

    logger.info(f"Detected face with bounding box: {largest_face['box']}")

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

    logger.info("Face region successfully cropped.")

    # Check the image quality of the extracted face
    quality_score, is_quality_sufficient = check_image_quality(face_image, logger)

    # Return the cropped face region and the quality information
    return {
        "face_image": face_image,
        "quality_score": quality_score,
        "is_quality_sufficient": is_quality_sufficient
    }
