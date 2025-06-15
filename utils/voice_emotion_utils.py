import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = "models/voice_emotion_model.h5"
model = load_model(model_path)

# Load the same label encoder used during training
EMOTIONS = ['01', '02', '03', '04', '05', '06', '07', '08']  # Based on RAVDESS codes

label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)

# Feature extraction
def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Record from microphone
def record_audio(filename='temp.wav', duration=3, fs=22050):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)
    print("✅ Recording finished.")
    return filename

# Predict emotion from audio file
def predict_emotion(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    features = extract_features(audio, sr)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label
