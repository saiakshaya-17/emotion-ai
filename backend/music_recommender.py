import pandas as pd

# =========================
# Load dataset ONCE
# =========================
music_df = pd.read_csv("Music Info.csv")

# =========================
# Normalize emotion labels
# =========================
EMOTION_NORMALIZE = {
    "angry": "anger",
    "anger": "anger",

    "sad": "sadness",
    "sadness": "sadness",

    "happy": "joy",
    "joy": "joy",

    "fear": "fear",
    "fearful": "fear",

    "surprise": "surprise",
    "surprised": "surprise",

    "neutral": "neutral"
}

# =========================
# Emotion → Genre mapping
# =========================
EMOTION_GENRE_MAP = {
    "anger": ["Rock", "Metal", "Punk"],
    "sadness": ["Blues", "Folk", "Acoustic"],
    "joy": ["Pop", "Electronic", "Dance"],
    "fear": ["Ambient", "New Age"],
    "surprise": ["Electronic"],
    "neutral": []
}

# =========================
# Recommendation function
# =========================
def recommend_songs(emotion, top_n=5):
    emotion = str(emotion).lower().strip()
    emotion = EMOTION_NORMALIZE.get(emotion, "neutral")

    df = music_df.copy()

    # Clean missing values
    df["genre"] = df["genre"].fillna("Unknown")
    df["name"] = df["name"].fillna("Unknown")
    df["artist"] = df["artist"].fillna("Unknown")

    genres = EMOTION_GENRE_MAP.get(emotion, [])
    if genres:
        df = df[df["genre"].isin(genres)]

    if "valence" in df.columns and "energy" in df.columns:
        df = df.sort_values(by=["valence", "energy"], ascending=False)

    return [
        {
            "name": row["name"],
            "artist": row["artist"],
            "genre": row["genre"]
        }
        for _, row in df.head(top_n).iterrows()
    ]
