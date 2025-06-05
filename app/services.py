import openai
import os
from dotenv import load_dotenv
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from openai import OpenAI
import json
from pathlib import Path
import html
import re
from .ssml_builder import SSMLBuilder
import logging

# DOCKER IMPLEMENTATION: Conditional .env loading for container compatibility
if os.path.exists('.env'):
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def process_text(text):
    """Send text to OpenAI GPT and return response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Error with OpenAI API: {e}")
        return "I'm sorry, I couldn't process that."

def load_voice_settings():
    # DOCKER IMPLEMENTATION: Container-friendly path resolution for voice settings
    config_path = Path(__file__).parent.parent / 'config' / 'voice_settings.json'
    
    # DOCKER IMPLEMENTATION: Fallback path for container environment
    if not config_path.exists():
        config_path = Path('/app/config/voice_settings.json')
    
    # DOCKER IMPLEMENTATION: Default settings if file not found in container
    if not config_path.exists():
        logging.warning("DOCKER IMPLEMENTATION: voice_settings.json not found, using defaults")
        return {
            "language_code": "fr-FR",
            "ssml_gender": "MALE",
            "speaking_rate": 1.0,
            "pitch": 0.0
        }
    
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"DOCKER IMPLEMENTATION: Error loading voice settings: {e}")
        return {
            "language_code": "fr-FR",
            "ssml_gender": "MALE", 
            "speaking_rate": 1.0,
            "pitch": 0.0
        }

VOICE_SETTINGS = load_voice_settings()

def text_to_speech(text, voice_settings):
    from google.cloud import texttospeech
    logger = logging.getLogger(__name__)
    try:
        ssml_text = add_natural_pauses(text)
        input_text = texttospeech.SynthesisInput(ssml=ssml_text)
    except Exception as e:
        logger.error(f"Error generating SSML: {e}")
        input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_settings['language_code'],
        name=voice_settings.get('voice_name', "fr-FR-Neural2-D"),
        ssml_gender=texttospeech.SsmlVoiceGender[voice_settings['ssml_gender']]
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=voice_settings['speaking_rate'],
        pitch=voice_settings['pitch']
    )
    client = texttospeech.TextToSpeechClient()
    try:
        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        logger.debug(f"audio_content type before return: {type(response.audio_content)}")
        if not isinstance(response.audio_content, bytes):
            raise RuntimeError("Google TTS did not return audio bytes!")
        return response.audio_content
    except Exception as e:
        logger.error(f"Error in Google TTS API: {e}")
        raise

def add_natural_pauses(text):
    # (Your existing SSML logic here)
    builder = SSMLBuilder()
    builder.add_text(text)
    return builder.to_ssml()

def speech_to_text(audio_content, config=None):
    client = speech.SpeechClient()
    if config is None:
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="fr-FR",
            # Add diarization or other options here if needed
        )
    audio = speech.RecognitionAudio(content=audio_content)
    response = client.recognize(config=config, audio=audio)
    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript
    return transcription

def chat_completion(messages, openai_key=None):
    client = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        temperature=0.7,
        top_p=1.0
    )
    return response.choices[0].message.content
