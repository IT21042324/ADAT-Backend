import tensorflow as tf
from keras import Model
from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, AveragePooling2D, \
    BatchNormalization, Dropout
from PIL import ImageFile
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

ImageFile.LOAD_TRUNCATED_IMAGES = True
from vit_keras import vit
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121

plt.switch_backend('agg')

# ===========================================================================

'''InceptionV3'''


def dens_create_plant_model(final_layer):
    base_model = DenseNet121(input_shape=(224, 224, 3),  # Shape of our images
                             include_top=False,  # Leave out the last fully connected layer
                             weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(base_model.output)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)

    # Add a final sigmoid layer for classification
    x = layers.Dense(final_layer, activation='softmax')(x)

    model = tf.keras.models.Model(base_model.input, x)

    return model


def InceptionV3_create_plant_model(final_layer):
    # Load InceptionV3 as the base model
    base_model = InceptionV3(input_shape=(299, 299, 3),
                             include_top=False,
                             weights='imagenet')

    # Make all layers untrainable by freezing weights (except for the last 15 layers)
    for layer in base_model.layers[:-15]:
        layer.trainable = False

    # Add Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Add a batch normalization layer
    x = layers.BatchNormalization()(x)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout layer with a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)

    # Add another fully connected layer with 256 hidden units and ReLU activation
    x = layers.Dense(256, activation='relu')(x)

    # Add another batch normalization layer
    x = layers.BatchNormalization()(x)

    # Add another dropout layer with a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)

    # Add the output layer with a softmax activation function for multi-class classification
    x = layers.Dense(final_layer, activation='softmax')(x)

    # Define the model by specifying the input and output
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    return model


'''ViT'''
# def vit_create_plant_model(final_layer):
#     vit_model = vit.vit_b16(
#         image_size=224,
#         activation='softmax',
#         pretrained=True,
#         include_top=False,
#         pretrained_top=False)
#
#     model = tf.keras.Sequential([
#         vit_model,
#         Flatten(),
#         Dense(final_layer, activation='softmax')
#     ])
#
#     return model



#
# def vit_create_plant_model(final_layer):
#     vit_model = vit.vit_b16(
#         image_size=384,
#         activation='softmax',
#         pretrained=True,
#         include_top=False,
#         pretrained_top=False)
#
#     model = tf.keras.Sequential([
#         vit_model,
#         Flatten(),
#         tf.keras.layers.Dense(128, activation ='relu'),
#         Dense(final_layer, activation='softmax')
#     ])
#     return model