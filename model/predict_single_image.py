import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_model.h5")
IMAGE_PATH = os.path.join(BASE_DIR, "..", "test_images", "test_leaf.jpg")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# --------------------------------------------------
# Predict
# --------------------------------------------------
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)

# --------------------------------------------------
# Class labels (IMPORTANT)
# Must match train_data.class_indices
# --------------------------------------------------
class_labels = ['Pepper_bell', 'Potato']

predicted_class = class_labels[predicted_class_index]
confidence = predictions[0][predicted_class_index] * 100

print("\n🌱 Prediction Result")
print("---------------------")
print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence:.2f}%")
