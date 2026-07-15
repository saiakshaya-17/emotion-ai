from flask import Flask, request, jsonify

from pipeline1.emotion_router import detect_emotion
from music_recommender import recommend_songs
from full_pipeline import full_pipeline

app = Flask(__name__)

@app.route("/")
def home():
    return "Emotion AI Backend is running"

# -----------------------------
# 1️⃣ Detect Emotion
# -----------------------------
@app.route("/detect-emotion", methods=["POST"])
def detect_emotion_api():
    data = request.json

    input_type = data.get("type")   # "text" or "face"
    payload = data.get("data")

    if not input_type or payload is None:
        return jsonify({"error": "Invalid request"}), 400

    emotion = detect_emotion(input_type, payload)
    return jsonify({"emotion": emotion})

# -----------------------------
# 2️⃣ Recommend Songs
# -----------------------------
@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.json
    emotion = data.get("emotion")

    if not emotion:
        return jsonify({"error": "Emotion is required"}), 400

    songs = recommend_songs(emotion)
    return jsonify({"emotion": emotion, "songs": songs})

# -----------------------------
# 3️⃣ Full Pipeline
# -----------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS
from full_pipeline import full_pipeline

app = Flask(__name__)
CORS(app)  # IMPORTANT

@app.route("/full-pipeline", methods=["POST"])
def run_full_pipeline():
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        input_type = data.get("input_type")
        payload = data.get("data")

        if not input_type or payload is None:
            return jsonify({"error": "Missing input_type or data"}), 400

        result = full_pipeline(input_type, payload)
        return jsonify(result)

    except Exception as e:
        print("🔥 BACKEND ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
