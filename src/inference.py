"""
Minimal inference: load model.keras, preprocess one image, return prediction.
"""
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from .preprocessing import load_image, resize_image, preprocess, add_batch


def predict_image(model, image_path):
    img = load_image(image_path)
    img = resize_image(img)
    img = preprocess(img)
    img = add_batch(img)

    preds = model.predict(img)
    # print(preds)
    class_id = np.argmax(preds, axis=1)[0]
    # print(class_id)
    confidence = preds[0][class_id]

    return class_id, confidence

# Example usage
# from .config import DATA_DIR
# 
# class_id, confidence = predict_image(
#     model,
#     str(DATA_DIR / "3.jpg")
# )
# 
# print("Predicted class:", class_id)
# print("Confidence:", confidence)
