from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras import optimizers
from tensorflow.python.keras.models import load_model
from keras import backend as K
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist
from tensorflow.keras import layers, models
from os.path import join, exists
import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np
import cv2


def load_image_with_preprocessing(img_path, input_size, show=False):
    img = image.load_img(img_path, target_size=input_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img, img_tensor


batch_size = 16


def generate_feature_masks(input_size, last_conv_layer_output, scale):
    cell_size = np.ceil(np.array(input_size) / scale)
    up_size = (scale) * cell_size
    grid = np.rollaxis(last_conv_layer_output, 2, 0)

    N = len(grid)
    masks = np.empty((*input_size, N))
    for i in range(N):
        feature_map = last_conv_layer_output[:, :, i]
        binary_mask = feature_map > 0.1
        binary_mask = binary_mask.astype('float32')
        resized_mask = resize(binary_mask, up_size, order=1, mode='reflect', anti_aliasing=False)
        masks[:, :, i] = resized_mask

    return masks, grid, cell_size, up_size


def calculate_similarity_weights(differences, kernel_width):
    return np.sqrt(np.exp(-(differences ** 2) / kernel_width ** 2))


def compute_similarity_differences(original_prediction, predictions):
    differences = abs(original_prediction - predictions)
    weights = calculate_similarity_weights(differences, 0.25)
    return weights, differences


def normalize_values(array):
    return (array - array.min()) / (array.max() - array.min() + 1e-13)


def compute_uniqueness(masks_predictions):
    sum_distances = (cdist(masks_predictions, masks_predictions)).sum(axis=1)
    normalized_distances = normalize_values(sum_distances)
    return normalized_distances


def compute_explanation(model, input_img, mask_count, p1, masks, input_size):
    preds = []
    masked_input = input_img * masks
    original_prediction = model.predict(input_img)

    for i in range(0, mask_count, batch_size):
        preds.append(model.predict(masked_input[i:min(i + batch_size, mask_count)]))
    preds = np.concatenate(preds)

    weights, differences = compute_similarity_differences(original_prediction, preds)
    uniqueness = compute_uniqueness(preds)
    reshaped_uniqueness = uniqueness.reshape(-1, 1)
    weighted_diff = np.multiply(weights, reshaped_uniqueness)

    saliency = weighted_diff.T.dot(masks.reshape(mask_count, -1)).reshape(-1, *input_size)
    saliency = saliency / mask_count / p1
    return saliency, weights, reshaped_uniqueness, weighted_diff, original_prediction


def build_model(final_layer):
    inception_resnet = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax",
    )
    conv_layer = inception_resnet.layers[-28].output
    activated_conv = Activation('relu')(conv_layer)
    dropout_layer = Dropout(0.5)(activated_conv)
    flattened_output = Flatten()(dropout_layer)
    final_output = Dense(final_layer, activation='softmax')(flattened_output)
    new_model = Model(inputs=inception_resnet.input, outputs=final_output)
    return new_model


base_model = build_model(6)
base_model.load_weights("model/model.hdf5")
base_model.summary()

# Find the last BatchNormalization layer
last_bn_layer = None
for layer in base_model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        last_bn_layer = layer
        break

if last_bn_layer:
    print("Last Batch Normalization Layer:", last_bn_layer.name)
else:
    print("No Batch Normalization layer found in the model.")

features_model = Model(inputs=base_model.input, outputs=base_model.get_layer(last_bn_layer.name).output)


def display_SIDU_heatmap(img_array, heatmap, pred_class):
    base_img = (img_array[0] + 1) / 2
    plt.imshow(base_img, alpha=1.0)
    plt.axis('off')
    colormap = 'Greens' if pred_class == 'normal' else 'jet'
    plt.imshow(cv2.resize(heatmap, (299, 299)), cmap=colormap, alpha=0.35)


target_names = ['Cyst', 'Pustules', 'black & white heads', 'normal', 'papules and nodules', 'scars']


def process_and_visualize(read_path):
    img, img_tensor = load_image_with_preprocessing(read_path, (299, 299))
    activation_maps = features_model.predict(img_tensor)
    last_conv_output = np.squeeze(activation_maps)
    masks, grid, cell_size, up_size = generate_feature_masks((299, 299), last_conv_output, scale=1)
    mask_array = np.rollaxis(masks, 2, 0)
    size = mask_array.shape
    reshaped_masks = mask_array.reshape(size[0], size[1], size[2], 1)
    masked_input = img_tensor * reshaped_masks
    N = len(mask_array)
    saliency, weights, interactions, differences, original_pred = compute_explanation(base_model, img_tensor, N, 0.5,
                                                                                      reshaped_masks, (299, 299))
    predicted_vector = base_model.predict(img_tensor)
    predicted_index = np.argmax(predicted_vector)
    predicted_class = target_names[predicted_index]
    plt.imshow(img)
    plt.imshow(saliency[predicted_index], cmap='jet', alpha=0.3)
    plt.axis('off')
    plt.show()


# Assuming load_image_with_preprocessing, features_model, base_model, and other dependencies are already defined

def display_SIDU_heatmap(img_array, heatmap, pred_class):
    base_img = (img_array[0] + 1) / 2
    plt.imshow(base_img, alpha=1.0)
    plt.axis('off')
    colormap = 'Greens' if pred_class == 'normal' else 'jet'
    plt.imshow(cv2.resize(heatmap, (299, 299)), cmap=colormap, alpha=0.35)


target_names = ['Cyst', 'Pustules', 'black & white heads', 'normal', 'papules and nodules', 'scars']


def process_and_visualize(read_path):
    img, img_tensor = load_image_with_preprocessing(read_path, (299, 299))
    activation_maps = features_model.predict(img_tensor)
    last_conv_output = np.squeeze(activation_maps)
    masks, grid, cell_size, up_size = generate_feature_masks((299, 299), last_conv_output, scale=1)
    mask_array = np.rollaxis(masks, 2, 0)
    size = mask_array.shape
    reshaped_masks = mask_array.reshape(size[0], size[1], size[2], 1)
    masked_input = img_tensor * reshaped_masks
    N = len(mask_array)
    saliency, weights, interactions, differences, original_pred = compute_explanation(base_model, img_tensor, N, 0.5,
                                                                                      reshaped_masks, (299, 299))
    predicted_vector = base_model.predict(img_tensor)
    predicted_index = np.argmax(predicted_vector)
    predicted_class_x = target_names[predicted_index]

    predicted_vector = np.squeeze(predicted_vector)
    top_3_indices = np.argsort(predicted_vector)[-3:][::-1]
    predicted_classes = [target_names[int(index)] for index in top_3_indices]
    second_predicted_class = predicted_classes[1]
    third_predicted_class = predicted_classes[2] if len(predicted_classes) > 2 else None

    # Create a buffer to store the image
    buf = BytesIO()

    # # Plot and save the image to the buffer
    plt.imshow(img)
    plt.imshow(saliency[predicted_index], cmap='jet', alpha=0.3)
    plt.axis('off')
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Move to the beginning of the buffer
    buf.seek(0)

    # Convert buffer to an image
    image = Image.open(buf)

    return image, predicted_class_x, second_predicted_class, third_predicted_class