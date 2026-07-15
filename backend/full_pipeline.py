from pipeline1.emotion_router import detect_emotion
from music_recommender import recommend_songs

def full_pipeline(input_type, data):
    """
    input_type: 'text', 'face', or 'both'
    data:
      - text → string
      - face → grayscale image (48x48)
      - both → { "face": img, "text": string }
    """

    emotion = detect_emotion(input_type, data)
    songs = recommend_songs(emotion)

    return {
        "emotion": emotion,
        "songs": songs
    }
