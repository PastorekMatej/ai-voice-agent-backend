from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Request
from openai import OpenAI, AuthenticationError
import asyncio
import speech_recognition as sr
import pyttsx3
import ssl
import certifi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
import json
from google.cloud import speech_v1p1beta1 as speech
import io
from pydub import AudioSegment
import tempfile
import logging
import subprocess
from pathlib import Path
from google.cloud import texttospeech
from fastapi.responses import FileResponse
import html
import re
from .ssml_builder import SSMLBuilder
from .performance import measure_performance, PerformanceTracker
import time
from .services import (
    load_voice_settings, text_to_speech, add_natural_pauses,
    speech_to_text, chat_completion
)
import traceback


# Configuration du logger au début du fichier
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configurer le chemin vers le fichier .env parent
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Définir directement le chemin des credentials sans passer par la variable d'environnement
credentials_path = r"C:\Users\pasto\ai-voice-agent\backend\ai-voice-agent-451616-5ab9c7176a3d.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
if not os.path.exists(credentials_path):
    raise ValueError(f"Credentials file not found at: {credentials_path}")

# Vérification des variables d'environnement
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(
    api_key=openai_key
)

# Test the API key with a simple request
try:
    test_response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using 3.5 for testing as it's cheaper
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("API Key verified successfully!")
except AuthenticationError as e:
    print(f"API Key verification failed: {str(e)}")
    raise ValueError("Invalid API key. Please check your OpenAI API key.")
except Exception as e:
    print(f"Unexpected error during API key verification: {str(e)}")
    raise

# Ensure SSL works correctly
ssl_context = ssl.create_default_context(cafile=certifi.where())

engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(f"ID: {voice.id}, Name: {voice.name}")

# Add this class for request validation
class ChatMessage(BaseModel):
    message: str

voice_settings = load_voice_settings()

@measure_performance("ssml")
def add_natural_pauses(text):
    """Convertit le texte en SSML avec expressivité et pauses naturelles."""
    try:
        with PerformanceTracker("ssml", "preprocessing", {"text_length": len(text)}):
        # Prétraitement
            logger.debug(f"Texte original: {text}")
            text = re.sub(r'([.!?])([^\s])', r'\1 \2', text)
            escaped_text = html.escape(text)
        
            # Initialiser le constructeur SSML
            builder = SSMLBuilder()
            
            # Division en phrases
            sentences = re.split(r'([.!?](?:\s|$))', escaped_text)
            logger.debug(f"Phrases détectées: {sentences}")
        
        # Process sentences and build SSML
        with PerformanceTracker("ssml", "building", {"sentences": len(sentences)}):
            for i in range(0, len(sentences), 2):
                if i >= len(sentences):
                    break
                    
                sentence = sentences[i]
                punctuation = sentences[i+1] if i+1 < len(sentences) else ""
                
                # Vérifier si la phrase est vide
                if not sentence.strip():
                    if punctuation:
                        builder.elements.append(punctuation)
                    continue
                
                # Traiter les mots à mettre en évidence
                sentence_with_emphasis = re.sub(r'\*(.*?)\*', 
                                            r'<emphasis level="moderate">\1</emphasis>', 
                                            sentence)
                
                # Ajouter des pauses pour les virgules dans tous les cas
                sentence_with_emphasis = re.sub(r',(\s)', r',<break time="350ms"/>\1', sentence_with_emphasis)
                
                # Traiter les mots importants
                important_words = ['dois', 'peux', 'veux', 'important', 'essentiel', 
                                'critique', 'nouveau', 'meilleur', 'pire', 'attention',
                                'nécessaire', 'crucial', 'clé', 'fondamental', 'merci', 'voilà']
                
                for word in important_words:
                    if word in sentence.lower():
                        pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
                        sentence_with_emphasis = pattern.sub(
                            f'<emphasis level="moderate">{word}</emphasis>', 
                            sentence_with_emphasis
                        )
                
                # Ajouter des variations UNIQUEMENT pour les questions et exclamations
                if punctuation and '?' in punctuation:
                    # Seulement légèrement plus aigu pour les questions
                    builder.elements.append(f'<prosody pitch="+5%">{sentence_with_emphasis}</prosody>')
                elif punctuation and '!' in punctuation:
                    # Légèrement plus fort pour les exclamations, mais pas plus rapide
                    builder.elements.append(f'<prosody volume="+5%">{sentence_with_emphasis}</prosody>')
                else:
                    # Laisser les phrases normales sans modification de vitesse
                    builder.elements.append(sentence_with_emphasis)
                
                # Ajouter la ponctuation avec pause appropriée
                if punctuation:
                    if '.' in punctuation:
                        builder.elements.append(f'{punctuation}<break time="750ms"/>')
                    elif '!' in punctuation:
                        builder.elements.append(f'{punctuation}<break time="800ms"/>')
                    elif '?' in punctuation:
                        builder.elements.append(f'{punctuation}<break time="750ms"/>')
                    else:
                        builder.elements.append(punctuation)
                
                # Respiration entre phrases
                builder.elements.append('<break time="100ms"/>')
            
        # Finalize SSML
        with PerformanceTracker("ssml", "finalization"):
            final_ssml = f'<speak>{" ".join(builder.elements)}</speak>'
            logger.debug(f"SSML final: {final_ssml}")
            
        return final_ssml
        
    except Exception as e:
        logger.error(f"Error building expressive SSML: {str(e)}")
        logger.error(f"Texte qui a causé l'erreur: {text}")
        return f"<speak>{html.escape(text)}</speak>"

@measure_performance("stt")
def speech_to_text(audio_file):
    """Converts speech (audio file) to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

@app.get("/")
async def root():
    """Test endpoint to verify server is running."""
    return {"status": "Server is running"}

@app.post("/api/chat")
async def chat(request: Request, message: ChatMessage):
    """Handles HTTP POST requests for chat."""
    logger.debug(f"Received request with message: {message.message}")
    try:
        logger.debug("Creating chat completion...")
        
        with PerformanceTracker("chat", "setup", {"message_length": len(message.message)}):
            # Récupérer les paramètres de langue et de personnalité
            language_level = voice_settings.get('language_level', 'B1')
            personality = voice_settings.get('personality', {})
            system_prompt = personality.get('system_prompt', "Tu es un assistant qui aide à apprendre le français.")
            adaptation_dynamique = personality.get('adaptation_dynamique', False)
            
            # Définir les descriptions de niveau
            level_descriptions = {
                "A2": "utilisant un vocabulaire simple et des phrases courtes",
                "B1": "utilisant un vocabulaire intermédiaire et des phrases relativement simples",
                "B2": "utilisant un vocabulaire courant et des phrases de complexité moyenne",
                "C1": "utilisant un vocabulaire riche et des phrases complexes",
                "C2": "utilisant un vocabulaire très riche, des expressions idiomatiques et des structures complexes"
            }
            
            level_instruction = level_descriptions.get(language_level, level_descriptions["B1"])
            
            # Construire les messages pour le chat
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Ajouter des instructions spécifiques au niveau si l'adaptation dynamique est activée
            if adaptation_dynamique:
                level_specific_instructions = f"L'élève est actuellement au niveau {language_level} du CECRL. Adapte ton vocabulaire et ta complexité grammaticale en conséquence, {level_instruction}. Ne corrige pas directement les erreurs, mais reformule correctement pour montrer le bon exemple."
                messages.append({"role": "system", "content": level_specific_instructions})
            
            # Ajouter le message de l'utilisateur
            messages.append({"role": "user", "content": message.message})
            
            # Récupérer les paramètres du modèle
            model_params = voice_settings.get('model_parameters', {})
            temperature = model_params.get('temperature', 0.7)
            top_p = model_params.get('top_p', 1.0)
            frequency_penalty = model_params.get('frequency_penalty', 0)
            presence_penalty = model_params.get('presence_penalty', 0)
            
            # Track API call
        with PerformanceTracker("chat", "openai_api_call"):
            # Appel à l'API avec les paramètres configurés
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
        
        logger.debug("Got response from OpenAI")
        ai_text = response.choices[0].message.content
        logger.debug(f"AI response: {ai_text}")
        
        return {"response": ai_text}
    except AuthenticationError as e:
        logger.error(f"Authentication Error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"OpenAI API Authentication Error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received WebSocket message: {data}")
            
            with PerformanceTracker("websocket", "openai_call"):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": data}]
                )
                
            ai_text = response.choices[0].message.content
            print(f"Sending WebSocket response: {ai_text}")
            await websocket.send_text(ai_text)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.post("/api/transcribe")
async def transcribe_audio(request: Request, audio: UploadFile = File(...)):
    try:
        # File processing
        with PerformanceTracker("transcribe", "file_processing"):
            # Vérifier que les credentials Google sont configurés
            if not credentials_path:
                    raise Exception("Google Cloud credentials not configured")
        
        # Créer un fichier temporaire pour l'audio WebM
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_path = webm_file.name
            # Écrire le contenu audio dans le fichier
            webm_file.write(await audio.read())
        
        try:
            # Audio conversion
            with PerformanceTracker("transcribe", "audio_conversion"):
                # Créer un fichier temporaire pour la sortie WAV
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                    wav_path = wav_file.name
                
                    # Utiliser FFmpeg pour la conversion
                    ffmpeg_cmd = [
                        r"C:\Users\pasto\ffmpeg\ffmpeg.exe",
                        '-y',
                        '-i', webm_path,
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        '-f', 'wav',
                        wav_path
                    ]
                
                    process = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True,
                        shell=True
                    )
                
            if process.returncode != 0:
                raise Exception(f"FFmpeg conversion failed with code {process.returncode}")
            
            # Prepare for speech recognition
            with open(wav_path, 'rb') as wav_file:
                converted_audio_content = wav_file.read()
            
            client = speech.SpeechClient()
            audio = speech.RecognitionAudio(content=converted_audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="fr-FR",  # Set to French
            )
            
            # Speech recognition API call
            with PerformanceTracker("transcribe", "google_api_call"):
                response = client.recognize(config=config, audio=audio)
                
            # Cleanup and process results
            os.unlink(webm_path)
            os.unlink(wav_path)
            
            transcription = ""
            for result in response.results:
                transcription += result.alternatives[0].transcript
            
            return {"transcription": transcription}
            
        except Exception as e:
            # Existing error handling
            raise HTTPException(status_code=422, detail=f"Error processing audio: {str(e)}")
            
    except Exception as e:
        # Existing error handling
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/api/synthesize")
async def synthesize_text(request: Request, message: ChatMessage):
    """Handles HTTP POST requests for text-to-speech synthesis."""
    logger.debug(f"Received request for TTS with message: {message.message}")
    try:
        with PerformanceTracker("synthesize", "tts_processing"):
            audio_content = text_to_speech(message.message, voice_settings)
        logger.debug(f"type(audio_content): {type(audio_content)}")  # Pour debug
        with open("output.mp3", "wb") as out:
            out.write(audio_content)
        response = FileResponse("output.mp3", media_type='audio/mpeg', filename='output.mp3')
        return response
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Entry point for running with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
