from pipeline1.text_emotion import predict_text_emotion
from pipeline1.face_emotion import predict_face_emotion

def detect_emotion(input_type, data):
    """
    input_type: 'text', 'face', or 'both'

    data:
      - if input_type == 'text':
            data = string
      - if input_type == 'face':
            data = grayscale image (48x48)
      - if input_type == 'both':
            data = {
                "face": grayscale image (48x48),
                "text": string
            }

    Rule:
      Face emotion DOMINATES text emotion
    """

    input_type = input_type.lower()

    if input_type == "face":
        return predict_face_emotion(data)

    elif input_type == "text":
        return predict_text_emotion(data)

    elif input_type == "both":
        # Face dominates
        face_img = data.get("face")
        if face_img is not None:
            return predict_face_emotion(face_img)

        # fallback to text if face missing
        text = data.get("text")
        return predict_text_emotion(text)

    else:
        raise ValueError("input_type must be 'text', 'face', or 'both'")
