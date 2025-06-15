# app/utils/face_emotion.py
import cv2
from fer import FER
import tempfile

def detect_face_emotion(video_file):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video_file.read())
    cap = cv2.VideoCapture(temp.name)

    detector = FER()
    emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect_emotions(frame)
        if results:
            emotions.append(results[0]['emotions'])
    
    cap.release()

    if emotions:
        # Return dominant emotion
        avg_emotion = max(emotions[-1], key=emotions[-1].get)
        return avg_emotion
    return "neutral"
