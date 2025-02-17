import openai
from app.config import OPENAI_API_KEY

def transcribe_audio(audio_file):
    response = openai.Audio.transcribe("whisper-1", audio_file, api_key=OPENAI_API_KEY)
    return response["text"]

def synthesize_voice(text):
    response = openai.Audio.create_speech(text=text, model="tts-1", voice="alloy", api_key=OPENAI_API_KEY)
    return response["url"]