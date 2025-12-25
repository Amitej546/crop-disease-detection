from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# --------------------------------------------------
# Load models ONCE (important)
# --------------------------------------------------
crop_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "crop_disease_model.h5")
)

potato_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "potato_disease_model.h5")
)

pepper_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "pepper_disease_model.h5")
)

print("✅ All models loaded")

# --------------------------------------------------
# Class label mappings (FIXED & SAFE)
# --------------------------------------------------
CROP_LABELS = {
    0: "Pepper_bell",
    1: "Potato"
}

POTATO_DISEASE_LABELS = {
    0: "Early_blight",
    1: "Late_blight",
    2: "healthy"
}

PEPPER_DISEASE_LABELS = {
    0: "Bacterial_spot",
    1: "healthy"
}

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    temp_path = "temp.jpg"
    file.save(temp_path)

    img_array = preprocess_image(temp_path)

    # -------- Stage 1: Crop --------
    crop_preds = crop_model.predict(img_array)
    crop_index = int(np.argmax(crop_preds))
    crop_conf = float(crop_preds[0][crop_index]) * 100
    crop_name = CROP_LABELS[crop_index]

    # -------- Stage 2: Disease --------
    if crop_name == "Potato":
        disease_preds = potato_model.predict(img_array)
        disease_index = int(np.argmax(disease_preds))
        disease_conf = float(disease_preds[0][disease_index]) * 100
        disease_name = POTATO_DISEASE_LABELS[disease_index]
    else:
        disease_preds = pepper_model.predict(img_array)
        disease_index = int(np.argmax(disease_preds))
        disease_conf = float(disease_preds[0][disease_index]) * 100
        disease_name = PEPPER_DISEASE_LABELS[disease_index]

    os.remove(temp_path)

    # -------- Final JSON --------
    return jsonify({
        "crop": crop_name,
        "crop_confidence": round(crop_conf, 2),
        "disease": disease_name,
        "disease_confidence": round(disease_conf, 2)
    })

# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
