from utils.voice_emotion_utils import record_audio, predict_emotion

file = record_audio()
emotion_code = predict_emotion(file)

emotion_map = {
    '01': "Neutral",
    '02': "Calm",
    '03': "Happy",
    '04': "Sad",
    '05': "Angry",
    '06': "Fearful",
    '07': "Disgust",
    '08': "Surprised"
}

print(f"🎭 Detected Emotion: {emotion_map.get(emotion_code, 'Unknown')}")
