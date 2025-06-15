# app/main.py
import streamlit as st
from app.utils.face_emotion import detect_face_emotion
from app.utils.voice_emotion import detect_voice_emotion
from backend.llm.emergency_llm import analyze_emergency

st.set_page_config(page_title="Emergency Analyzer", layout="wide")
st.title("🚨 Real-Time Multimodal Emergency Analyzer")

# Upload video/audio
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if st.button("Analyze Emergency"):
    if uploaded_video and uploaded_audio:
        face_emotion = detect_face_emotion(uploaded_video)
        voice_emotion = detect_voice_emotion(uploaded_audio)
        
        emergency_level = analyze_emergency(face_emotion, voice_emotion)
        
        st.success(f"🔍 Emergency Status: {emergency_level}")
    else:
        st.warning("Please upload both video and audio.")
