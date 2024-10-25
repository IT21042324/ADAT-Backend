# Standard library imports
import io
import base64

# Third-party imports
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# Project-specific imports (classifier module)
from classifier.ViT import load_entire_vit_model
from classifier.acne_classifier import weighted_vote_single_image
from classifier.classifier import predict_single_image
from classifier.face_extractor import extract_face
from classifier.resNext import load_resNext_model
from classifier.rf import load_rf_with_feature_extractor

def classify_acne_image(file, binary_classifier_model, rf_feature_extractor, rf_model, vit_model, resNext_model, logger):
    try:
        logger.info("Received a request for image classification")

        # Read the image bytes
        image_bytes = file.read()

        # Validate file size (5MB = 5 * 1024 * 1024 bytes)
        max_file_size = 5 * 1024 * 1024  # 5 MB
        if len(image_bytes) > max_file_size:
            logger.warning(f"File size exceeds limit: {len(image_bytes)} bytes")
            return {"message": "File size exceeds 5MB limit. Please upload a smaller file."}, 413

        # Validate file type
        supported_image_types = ['image/jpg', 'image/jpeg', 'image/png', 'image/webp']
        if file.content_type not in supported_image_types:
            logger.warning(f"Invalid file type: {file.content_type}")
            return {"message": "Invalid file type. Only JPEG, PNG, or WebP is supported"}, 400

        # Try to open the image using PIL, catch specific errors for corrupted or invalid images
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.info("Image opened successfully and converted to RGB")
        except UnidentifiedImageError:
            logger.error("Corrupted image file or unsupported format")
            return {"message": "Corrupted image file or unsupported format. Please upload a valid image."}, 400

        # Resize the image for face detection
        image = ImageOps.fit(image, (512, 512), Image.Resampling.LANCZOS)
        logger.info("Image resized to 512x512 for face detection")

        # Convert the PIL image to a NumPy array
        open_cv_image = np.array(image)
        logger.info("Image converted to NumPy array for further processing")

        # Extract the face using the face_extractor function
        try:
            face_data = extract_face(open_cv_image, logger= logger)
            face_image = face_data["face_image"]
            is_quality_sufficient = face_data["is_quality_sufficient"]
            logger.info("Face extracted successfully")
        except ValueError as ve:
            # Catch errors like "no face" and return them to the client
            logger.warning(f"Face detection error: {str(ve)}")
            return {"message": str(ve)}, 400

        # Convert the extracted face image back to a PIL image
        face_image_pil = Image.fromarray(face_image)

        # Save the face image to a buffer for base64 encoding
        buffered = io.BytesIO()
        face_image_pil.save(buffered, format="JPEG")
        face_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info("Face image encoded to base64")

        if not is_quality_sufficient:
            logger.warning("Face image quality insufficient for accurate classification.")
            return {
                "result": "Low Quality",
                "quality_score": face_data["quality_score"],
                "message": "The uploaded face image is of low quality and might not provide accurate classification results.",
                "face_image": face_image_str,  # Return the extracted face image
            }

        # Predict using the binary classifier model if quality is sufficient
        result, confidence = predict_single_image(face_image_pil, binary_classifier_model)
        logger.info(f"Binary Classifier Result: {result}, Confidence: {confidence}")

        # If the result is "Acne", further classify the types of acne
        if result == "Acne":
            acne_types, probabilities = weighted_vote_single_image(
                face_image_pil,
                rf_feature_extractor,
                rf_model,
                vit_model,
                resNext_model
            )

            logger.info(f"Acne Types: {acne_types}, Probabilities: {probabilities}")

            return {
                "result": result,
                "confidence": confidence,
                "acne_types": acne_types,
                "probabilities": probabilities,
                "face_image": face_image_str,  # Optionally return the extracted face image
            }

        # Return the result and confidence for non-acne images
        logger.info("Returning classification result for non-acne image")
        return {
            "result": result,
            "confidence": confidence,
            "face_image": face_image_str,  # Optionally return the extracted face image
        }

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        return {"message": "Model files are missing. Please contact the administrator."}, 500

    except Exception as e:
        # Log the unexpected error and return a generic error message
        logger.error(f"Unexpected error: {str(e)}")
        return {"message": "An error occurred while processing your request. Please try again."}, 400