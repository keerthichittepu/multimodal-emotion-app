import streamlit as st
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load models
voice_model = load_model("models/voice_emotion_model.h5")
face_model = load_model("models/face_emotion_model.h5")

# Emotion labels (adjust if different in your training)
voice_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
face_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def extract_mfcc(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

def predict_voice_emotion():
    st.info("🎙️ Recording... Speak now for 3 seconds.")
    audio = sd.rec(int(3 * 22050), samplerate=22050, channels=1)
    sd.wait()
    audio = audio.flatten()
    features = extract_mfcc(audio, 22050)
    prediction = voice_model.predict(features)
    emotion = voice_emotions[np.argmax(prediction)]
    return emotion

def predict_face_emotion():
    st.info("📷 Capturing image from webcam...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "No Face"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    prediction = face_model.predict(face)
    emotion = face_emotions[np.argmax(prediction)]
    return emotion

def is_emergency(voice_emotion, face_emotion):
    danger_emotions = ['angry', 'fearful', 'disgust', 'sad']
    return voice_emotion in danger_emotions and face_emotion in danger_emotions

st.title("🚨 Multimodal Emergency Analyzer")

if st.button("🔍 Analyze Emotions"):
    voice_emotion = predict_voice_emotion()
    st.success(f"🎤 Voice Emotion: **{voice_emotion}**")

    face_emotion = predict_face_emotion()
    st.success(f"🧑 Face Emotion: **{face_emotion}**")

    if is_emergency(voice_emotion, face_emotion):
        st.error("⚠️ Emergency Detected! Alert Triggered!")
    else:
        st.success("✅ No Emergency Detected.")
