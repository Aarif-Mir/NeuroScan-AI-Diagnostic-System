import os
import numpy as np
import cv2
import pydicom

def load_image(path):
    if path.lower().endswith(".dcm"):
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.float32)

        # Normalize DICOM intensity to 0–255
        img = img - img.min()
        img = img / (img.max() + 1e-6)
        img = (img * 255).astype(np.uint8)

        # Convert grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    else:
        img = cv2.imread(path)                 # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def resize_image(img, size=(224, 224)):
    return cv2.resize(img, size)

from keras.applications.resnet50 import preprocess_input

def preprocess(img):
    img = img.astype(np.float32)
    img = preprocess_input(img)   # RGB → BGR + mean subtraction
    return img


def add_batch(img):
    return np.expand_dims(img, axis=0)



