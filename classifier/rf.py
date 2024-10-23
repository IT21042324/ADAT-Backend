import os
import pickle
import numpy as np
import torch
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input

# Define the class names for acne types
class_names = ['comedones', 'cysts', 'nodules', 'papules', 'pustules']

# Function to determine the device to use (GPU or CPU)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Function to load the Random Forest feature extractor model
from tensorflow.keras.models import load_model, Model
import os
import pickle


# Function to load the feature extractor (Keras model)
def load_random_forest_feature_extractor():
    model_path = os.path.join('model', 'xception_feature_extractor_tf.h5')

    # Load the model without compiling to avoid optimizer issues
    feature_extractor_model = load_model(model_path, compile=False)
    # Extract the part of the model you need (before the final layers)
    feature_extractor = Model(inputs=feature_extractor_model.input,
                              outputs=feature_extractor_model.get_layer('global_average_pooling2d').output)
    return feature_extractor

# Function to load the trained Random Forest model
def load_rf_sigmoid():
    model_path = os.path.join('model', 'rf_model_with_xception_extraction.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Function to load both the feature extractor and the Random Forest model
def load_rf_with_feature_extractor():
    # Load feature extractor without compilation and the Random Forest model
    return load_random_forest_feature_extractor(), load_rf_sigmoid()

# Function to extract features using the feature extractor model
def extract_features_for_rf(x, feature_extractor):
    extracted_features = feature_extractor.predict(x)
    return extracted_features.reshape(extracted_features.shape[0], -1) if len(extracted_features.shape) > 2 else extracted_features

# Function to load and preprocess images
def load_and_preprocess_images_rf(image_names, image_dir, target_size=(299, 299)):
    images = []
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        images.append(image_array)
    return np.array(images)

# Function to preprocess the image data and extract features
def preprocess_image_and_extract_features_for_rf(df, dataset_dir, feature_extractor):
    preprocessed_data = load_and_preprocess_images_rf(df['image_name'].values, dataset_dir)
    return extract_features_for_rf(preprocessed_data, feature_extractor)

# Function to extract features from a single image using the feature extractor model
def extract_features_for_single_image_rf(image_array, feature_extractor):
    extracted_features = feature_extractor.predict(image_array)
    return extracted_features.reshape(1, -1) if len(extracted_features.shape) > 2 else extracted_features

# Function to convert sigmoid probabilities to softmax-like probabilities
def sigmoid_to_softmax(probabilities):
    exp_probs = np.exp(probabilities)
    softmax_probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
    return softmax_probs

# Function to predict the probabilities for a single image using the Random Forest model
def predict_single_image_rf(image_pil, feature_extractor, rf_model):
    # Resize the image to the expected size
    image_pil = image_pil.resize((299, 299))

    # Preprocess the single image
    image_array = img_to_array(image_pil)
    image_array = preprocess_input(image_array)
    preprocessed_image = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Extract features using the feature extractor
    extracted_features = extract_features_for_single_image_rf(preprocessed_image, feature_extractor)
    
    # Predict with the Random Forest model
    probabilities = rf_model.predict_proba(extracted_features)
    
    # Extract the probability for the positive class (class 1) for each label
    if isinstance(probabilities, list):
        probabilities = np.array([prob[:, 1] for prob in probabilities]).T  # Get the probability of class 1 for each label
    else:
        probabilities = probabilities[:, 1]  # In case of binary classification, get the positive class probabilities
    
    # Convert sigmoid probabilities to softmax-like probabilities
    softmax_probabilities = sigmoid_to_softmax(probabilities)
    
    return softmax_probabilities
