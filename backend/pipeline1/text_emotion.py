import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Resolve paths safely (NO hardcoding)
# -------------------------------------------------

# pipeline1/text_emotion.py → backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "text_emotion_model.h5")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# -------------------------------------------------
# Load model & assets (ONCE)
# -------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 40  # must match training

# -------------------------------------------------
# Text normalization (negation handling)
# -------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()

    negation_patterns = [
        "not happy",
        "should be happy but",
        "but i am not",
        "not feeling good",
        "nothing feels right",
        "feel empty",
        "feels pointless"
    ]

    for p in negation_patterns:
        if p in text:
            return "i feel sad"

    return text

# -------------------------------------------------
# Prediction function
# -------------------------------------------------

def predict_text_emotion(text: str) -> str:
    text = normalize_text(text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    preds = model.predict(padded, verbose=0)
    label_id = np.argmax(preds, axis=1)[0]

    return label_encoder.inverse_transform([label_id])[0]
