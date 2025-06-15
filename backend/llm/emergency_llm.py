# backend/llm/emergency_llm.py
from openai import OpenAI

client = OpenAI(api_key="your-openai-api-key")

def analyze_emergency(face_emotion, voice_emotion):
    prompt = f"""
    A person shows '{face_emotion}' on their face and sounds '{voice_emotion}' in their voice.
    Determine if this is an emergency situation. Respond with one of: ["No Emergency", "Mild", "Critical"].
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
