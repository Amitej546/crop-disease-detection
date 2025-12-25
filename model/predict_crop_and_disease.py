import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CROP_MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_model.h5")
POTATO_MODEL_PATH = os.path.join(BASE_DIR, "potato_disease_model.h5")
PEPPER_MODEL_PATH = os.path.join(BASE_DIR, "pepper_disease_model.h5")

IMAGE_PATH = os.path.join(BASE_DIR, "..", "test_images", "test_leaf.jpg")

# --------------------------------------------------
# Load models
# --------------------------------------------------
crop_model = tf.keras.models.load_model(CROP_MODEL_PATH)
potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH)
pepper_model = tf.keras.models.load_model(PEPPER_MODEL_PATH)

print("✅ All models loaded successfully")

# --------------------------------------------------
# Preprocess image
# --------------------------------------------------
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# --------------------------------------------------
# Stage 1: Crop Prediction
# --------------------------------------------------
crop_preds = crop_model.predict(img_array)
crop_index = np.argmax(crop_preds)
crop_confidence = crop_preds[0][crop_index] * 100

crop_labels = ["Pepper_bell", "Potato"]
predicted_crop = crop_labels[crop_index]

# --------------------------------------------------
# Stage 2: Disease Prediction
# --------------------------------------------------
if predicted_crop == "Potato":
    disease_preds = potato_model.predict(img_array)
    disease_labels = ["Early_blight", "Late_blight", "healthy"]
else:
    disease_preds = pepper_model.predict(img_array)
    disease_labels = ["Bacterial_spot", "healthy"]

disease_index = np.argmax(disease_preds)
disease_confidence = disease_preds[0][disease_index] * 100
predicted_disease = disease_labels[disease_index]

# --------------------------------------------------
# Final Output
# --------------------------------------------------
print("\n🌱 FINAL PREDICTION RESULT")
print("--------------------------")
print(f"Detected Crop    : {predicted_crop} ({crop_confidence:.2f}%)")
print(f"Detected Disease : {predicted_disease} ({disease_confidence:.2f}%)")
