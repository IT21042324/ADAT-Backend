import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Define the class names for acne types
class_names = ['comedones', 'cysts', 'nodules', 'papules', 'pustules']

# Function to determine the device to use (GPU or CPU)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Custom ResNeXt model definition
class CustomResNeXt(nn.Module):
    def __init__(self, base_model, hidden_units, dropout_rate, num_hidden_layers, num_classes):
        super(CustomResNeXt, self).__init__()

        # Extract all layers except the final fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # Get the number of input features for the first fully connected layer
        input_units = base_model.fc.in_features

        # Define the custom classifier
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_units, hidden_units))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            input_units = hidden_units

        layers.append(nn.Linear(hidden_units, num_classes))  # Output layer for num_classes
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)  # Pass through feature extractor (including avg pooling)
        x = torch.flatten(x, 1)  # Flatten to match the input of the classifier
        x = self.classifier(x)
        return x  # Sigmoid will be applied by the loss function

# Define the image transformations as expected by the ResNeXt model
def get_resNext_image_transformation():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Function to load the ResNeXt model from a file
def load_resNext_model():
    model_path = os.path.join('model', 'ResNeXt-101-32x8d_sigmoid_complete_model.pth')
    loaded_model = torch.load(model_path, map_location=device)
    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model

# Function to get softmax-like predictions for a single image using the ResNeXt model
def predict_single_image_with_resNext_softmax(model, image_pil):
    model.eval()

    # Preprocess the image (no need to load it from a file anymore)
    image = get_resNext_image_transformation()(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output).cpu().numpy()[0]

        # Convert to softmax-like probabilities
        softmax_like_probs = probabilities / np.sum(probabilities)
    return softmax_like_probs
