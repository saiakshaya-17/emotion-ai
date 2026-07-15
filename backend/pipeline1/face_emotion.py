import os
import cv2
import numpy as np
import tensorflow as tf
import base64

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "cam_emotion_model.h5")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ---------------- Load ----------------
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

FER_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# ---------------- Utils ----------------
def decode_base64_image(base64_str):
    header, encoded = base64_str.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    return img

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face

# ---------------- Main ----------------
def predict_face_emotion(base64_image: str) -> str:
    gray = decode_base64_image(base64_image)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48)
    )

    if len(faces) == 0:
        return "Neutral"  # SAFE fallback

    # take largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]

    processed = preprocess_face(face)
    preds = model.predict(processed, verbose=0)
    idx = int(np.argmax(preds))

    return FER_LABELS[idx]
