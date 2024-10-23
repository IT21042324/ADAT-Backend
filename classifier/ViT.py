import os
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from PIL import Image

# Define the class names
class_names = ['comedones', 'cysts', 'nodules', 'papules', 'pustules']

# Function to determine the device to use (GPU or CPU)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Define a custom model that adds dropout to the original ViT model
class ViTWithDropout(nn.Module):
    def __init__(self, original_vit):
        super().__init__()
        self.vit = original_vit
        self.dropout = nn.Dropout(0.3)  # Adds dropout

    def forward(self, x):
        x = self.vit(x)
        x = self.dropout(x)
        return x

# Define the image transformations as expected by the pretrained model
def get_ViT_image_transformation():
    return transforms.Compose([
        transforms.Resize(256),  # Resize the smallest side to 256
        transforms.CenterCrop(224),  # Crop the center to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

model_path = os.path.join('model', 'final_entire_model_v2.pth')

def load_entire_vit_model(model_path=model_path):
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return model

# Function to predict the class probabilities for a single image using the ViT model
def predict_single_image_for_vit_softmax(model, image_pil, class_names=class_names):
    model.eval()  # Ensure the model is in evaluation mode
    
    # Preprocess the image
    image = get_ViT_image_transformation()(image_pil).unsqueeze(0).to(device)  # Apply preprocessing, add batch dimension, and move to device
    
    # Make prediction
    with torch.no_grad():
        output = model(image)  # Forward pass through the model
        probabilities = torch.sigmoid(output).cpu().numpy()[0]  # Apply sigmoid for multi-label classification
        softmax_like_probs = probabilities / np.sum(probabilities)  # Convert to softmax-like probabilities
        
    return softmax_like_probs
