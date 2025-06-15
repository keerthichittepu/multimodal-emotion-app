# app/utils/voice_emotion.py
import librosa
import numpy as np
from keras.models import load_model
import tempfile

model = load_model("models/voice_emotion_model.h5")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def detect_voice_emotion(audio_file):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(audio_file.read())
    features = extract_features(temp.name)
    prediction = model.predict(np.expand_dims(features, axis=0))
    emotion = np.argmax(prediction)
    labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    return labels[emotion]
