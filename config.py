from enum import Enum

saved_image_directory = "runs/segment/predict/"
FINAL_LAYERS = [2, 3, 4]

diseases_model_class_names = ['Abnormal', 'Normal']

IMG_SIZE = (299, 299)
IMG_SIZE1 = (224, 224)
LAST_LAYER_BINARY = "conv5_block16_2_conv"
LAST_LAYER_DENS_BACK = "conv5_block16_2_conv"
THRESHOLD = 70
BATCH_SIZE = 16
Reccomentdation =  ['Immediate dermatological consultation is crucial to address the extremely severe acne condition and prevent complications such as scarring and inflammation.', 'Treatment options may include prescription medications like oral antibiotics, retinoids, or isotretinoin to target the underlying causes of severe acne.', 'In-office procedures such as corticosteroid injections or drainage of large cysts may be necessary for rapid improvement.', 'Lifestyle changes such as a gentle skincare routine, avoiding picking or squeezing lesions, and maintaining a healthy diet can support acne management.']
selfTreatments =  ['Over-the-counter topical treatments containing benzoyl peroxide or salicylic acid can help reduce inflammation and unclog pores.', 'Gentle cleansing with a mild, non-comedogenic cleanser twice daily can help maintain skin hygiene.', 'Use oil-free and non-comedogenic moisturizers to keep the skin hydrated without exacerbating acne.', 'Avoid excessive sun exposure and use non-comedogenic sunscreen to protect the skin.', 'Monitor for any signs of infection or worsening of lesions and seek immediate medical attention if needed.']

class SeverityBadge(Enum):
    RED = "Red"
    GREEN = "Green"
    YELLOW = "Yellow"


class Final_Condition_Binary(Enum):
    NOT_NORMAL = "Not-Normal"
    NORMAL = "Normal"


class Is_Pain(Enum):
    NO = 0
    YES = 1
