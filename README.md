🌱 Crop Disease Detection System (Two-Stage AI Pipeline)

An **AI-powered crop disease detection system** that uses **deep learning (CNNs)** and a **two-stage classification pipeline** to accurately identify the **crop type** and its **disease** from a leaf image.  
The system is deployed using a **Flask REST API** and is ready for web or mobile integration.

---

## 📌 Problem Statement

Plant diseases are responsible for **20–40% annual crop losses**, mainly due to late or incorrect detection.  
Manual inspection is time-consuming and often inaccurate.

This project aims to:
- Detect crop diseases early using **image recognition**
- Reduce misclassification between visually similar crops
- Provide a **scalable backend API** for real-world deployment

---

## 🚀 Solution Overview

We implemented a **two-stage deep learning pipeline**:

Leaf Image
↓
Stage 1: Crop Classification (Potato / Pepper_bell)
↓
Stage 2: Crop-Specific Disease Classification
↓
Final Output (Crop + Disease + Confidence)


This approach significantly **reduces inter-crop confusion**, which is common when using a single multi-class model.

---

## 🧠 System Architecture

### 🔹 Stage 1 – Crop Classifier
- Identifies the crop type from the leaf image
- Model: **MobileNetV2 (Transfer Learning)**

### 🔹 Stage 2 – Disease Classifier
- Uses a **crop-specific CNN**
- Potato → Early blight / Late blight / Healthy  
- Pepper_bell → Bacterial spot / Healthy  

### 🔹 Backend
- **Flask REST API**
- Accepts image uploads
- Returns JSON response with predictions

---

## 🧪 Sample API Output

```json
{
  "crop": "Potato",
  "crop_confidence": 99.99,
  "disease": "Early_blight",
  "disease_confidence": 100.0
}
🛠️ Technologies Used
🔹 Programming & Frameworks
Python 3

TensorFlow / Keras

Flask

🔹 Deep Learning
Convolutional Neural Networks (CNN)

Transfer Learning (MobileNetV2)

Data Augmentation

🔹 Tools
Git & GitHub

Postman (API testing)

VS Code

📂 Project Structure
bash
Copy code
crop-disease-detection/
│
├── backend/
│   └── app.py                  # Flask API
│
├── model/
│   ├── crop_disease_model.h5   # Stage-1 crop classifier
│   ├── potato_disease_model.h5 # Stage-2 potato diseases
│   ├── pepper_disease_model.h5 # Stage-2 pepper diseases
│   ├── train_model.py
│   ├── train_potato_disease_model.py
│   ├── train_pepper_disease_model.py
│   └── predict_crop_and_disease.py
│
├── requirements.txt
├── .gitignore
└── README.md
▶️ How to Run the Project
1️⃣ Clone the Repository

git clone https://github.com/Amitej546/crop-disease-detection.git
cd crop-disease-detection
2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies

pip install -r requirements.txt
4️⃣ Run Flask Backend

python backend/app.py
Server runs at:

http://127.0.0.1:5000
🧪 API Usage
Endpoint
bash
Copy code
POST /predict
Request
Content-Type: multipart/form-data

Key: image

Value: Leaf image file

🎯 Key Features
✅ Two-stage AI pipeline (Crop → Disease)

✅ High accuracy with transfer learning

✅ REST API for easy integration

✅ Modular & scalable architecture

✅ Ready for web or mobile apps

⚠️ Limitations
Trained mainly on PlantVillage-style images

Performance may vary on real-world field images due to domain shift

Remedy recommendation not yet automated

🔮 Future Scope
Add disease-specific remedies

Convert models to TensorFlow Lite for mobile apps

Improve robustness using real field images

Add frontend dashboard for farmers

👨‍💻 Author
Amitej Kasarla
GitHub: https://github.com/Amitej546

🌱 Early detection saves crops, effort, and livelihoods.