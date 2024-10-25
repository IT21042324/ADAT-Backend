import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from .rf import predict_single_image_rf, load_rf_with_feature_extractor
from .ViT import predict_single_image_for_vit_softmax, load_entire_vit_model
from .resNext import predict_single_image_with_resNext_softmax, load_resNext_model


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_classnames():
    return ['comedones', 'cysts', 'nodules', 'papules', 'pustules']


# Asynchronous wrapper for predicting with Random Forest
async def async_predict_rf(image_pil, rf_feature_extractor, rf_model):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, predict_single_image_rf, image_pil, rf_feature_extractor, rf_model)


# Asynchronous wrapper for predicting with ViT
async def async_predict_vit(vit_model, image_pil, class_names):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, predict_single_image_for_vit_softmax, vit_model, image_pil, class_names)


# Asynchronous wrapper for predicting with ResNeXt
async def async_predict_resnext(resnext_model, image_pil):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, predict_single_image_with_resNext_softmax, resnext_model, image_pil)


# Main function to gather predictions asynchronously and perform weighted voting
async def weighted_vote_single_image(image_pil, rf_feature_extractor, rf_model, vit_model, resnext_model,
                                     class_names=get_classnames()):
    # F1 scores for each model (used as weights)
    f1_scores_resnext = {'comedones': 0.87, 'cysts': 0.90, 'nodules': 0.89, 'papules': 0.89, 'pustules': 0.84}
    f1_scores_vit = {'comedones': 0.86, 'cysts': 0.92, 'nodules': 0.84, 'papules': 0.87, 'pustules': 0.75}
    f1_scores_rf = {'comedones': 0.90, 'cysts': 0.87, 'nodules': 0.80, 'papules': 0.88, 'pustules': 0.87}

    # Run predictions asynchronously in parallel
    rf_task = async_predict_rf(image_pil, rf_feature_extractor, rf_model)
    vit_task = async_predict_vit(vit_model, image_pil, class_names)
    resnext_task = async_predict_resnext(resnext_model, image_pil)

    # Wait for all model predictions to complete
    rf_softmax_probs, vit_softmax_probs, resnext_softmax_probs = await asyncio.gather(rf_task, vit_task, resnext_task)

    # Flatten Random Forest softmax probabilities
    rf_softmax_probs = rf_softmax_probs.flatten()

    # Initialize dictionaries to accumulate probabilities
    accumulated_probs = {cls: 0.0 for cls in class_names}

    # Add weighted Random Forest probabilities
    for i, cls in enumerate(class_names):
        accumulated_probs[cls] += rf_softmax_probs[i] * f1_scores_rf[cls]

    # Add weighted ViT probabilities
    for i, cls in enumerate(class_names):
        accumulated_probs[cls] += vit_softmax_probs[i] * f1_scores_vit[cls]

    # Add weighted ResNeXt probabilities
    for i, cls in enumerate(class_names):
        accumulated_probs[cls] += resnext_softmax_probs[i] * f1_scores_resnext[cls]

    # Normalize by the sum of weights for each class
    for cls in accumulated_probs:
        total_weight = f1_scores_resnext[cls] + f1_scores_rf[cls] + f1_scores_vit[cls]
        accumulated_probs[cls] = round(accumulated_probs[cls] / total_weight, 2)  # Round to 2 decimal places

    # Apply threshold to final probabilities
    final_classes = [cls for cls in accumulated_probs if accumulated_probs[cls] >= 0.0]
    final_probs = [accumulated_probs[cls] for cls in final_classes]

    return final_classes, final_probs
