from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
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
<<<<<<< HEAD
=======
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

# Configuration du logger au début du fichier
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not api_key.startswith('sk-'):
    raise ValueError("OPENAI_API_KEY seems invalid - should start with 'sk-'")

print(f"API Key length: {len(api_key)}")  # This will print the length without showing the key

# Initialize OpenAI client
client = OpenAI(api_key=api_key)
=======
# Configurer le chemin vers le fichier .env parent
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Vérification des variables d'environnement
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
openai_key = os.getenv("OPENAI_API_KEY")

if not credentials_path:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found in environment variables")
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
>>>>>>> 04f82b2 (new image feature)

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

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

engine = pyttsx3.init()

# Add this class for request validation
class ChatMessage(BaseModel):
    message: str

def text_to_speech(text):
    """Converts text to speech using Google Cloud Text-to-Speech."""
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Configure the voice request, select the language code ("fr-FR") and the ssml voice gender
    voice = texttospeech.VoiceSelectionParams(
        language_code="fr-FR",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected voice parameters and audio file type
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        logger.debug('Audio content written to file "output.mp3"')

    return "output.mp3"

def speech_to_text(audio_file):
    """Converts speech (audio file) to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    """Handles real-time WebSocket communication for voice chat."""
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()  # Adjust to receive binary data
        # Process the audio data here
        # Convert audio to text, get AI response, and send back text
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "audio data processed"}]
        )
        ai_text = response.choices[0].message.content
        text_to_speech(ai_text)
        await websocket.send_text(ai_text)

@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Handles HTTP POST requests for chat."""
    logger.debug(f"Received request with message: {message.message}")
    try:
        logger.debug("Creating chat completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message.message}],
            max_tokens=150  # Optionnel : limite la longueur de la réponse
        )
        logger.debug("Got response from OpenAI")
        ai_text = response.choices[0].message.content
        logger.debug(f"AI response: {ai_text}")
        return {"response": ai_text}
<<<<<<< HEAD
=======
    except AuthenticationError as e:
        logger.error(f"Authentication Error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"OpenAI API Authentication Error: {str(e)}"
        )
>>>>>>> 04f82b2 (new image feature)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/")
async def root():
    return {
        "message": "Hello, World!",
        "status": "Frontend is communicating with the backend."
    }

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Vérifier que les credentials Google sont configurés
        if not credentials_path:
            raise Exception("Google Cloud credentials not configured. Please set GOOGLE_APPLICATION_CREDENTIALS in .env file")
        
        logger.debug("Received audio file")
        
        # Créer un fichier temporaire pour l'audio WebM
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_path = webm_file.name
            # Écrire le contenu audio dans le fichier
            webm_file.write(await audio.read())
        
        logger.debug(f"Saved WebM to temporary file: {webm_path}")
        
        try:
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
            
            logger.debug(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            process = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                shell=True
            )
            
            if process.returncode != 0:
                logger.error(f"FFmpeg stderr: {process.stderr}")
                logger.error(f"FFmpeg stdout: {process.stdout}")
                raise Exception(f"FFmpeg conversion failed with code {process.returncode}")
            
            logger.debug("FFmpeg conversion successful")
            
            # Lire le fichier WAV converti
            with open(wav_path, 'rb') as wav_file:
                converted_audio_content = wav_file.read()
            
            logger.debug(f"WAV file size: {len(converted_audio_content)} bytes")
            
            # Configurer la reconnaissance vocale
            client = speech.SpeechClient()
            audio = speech.RecognitionAudio(content=converted_audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )
            
            # Effectuer la transcription
            logger.debug("Sending to Google Speech-to-Text")
            response = client.recognize(config=config, audio=audio)
            logger.debug(f"Got response: {response}")
            
            # Nettoyer les fichiers temporaires
            os.unlink(webm_path)
            os.unlink(wav_path)
            
            # Extraire le texte
            transcription = ""
            for result in response.results:
                transcription += result.alternatives[0].transcript
            
            return {"transcription": transcription}
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            # Nettoyer les fichiers temporaires en cas d'erreur
            if os.path.exists(webm_path):
                os.unlink(webm_path)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.unlink(wav_path)
            raise HTTPException(status_code=422, detail=f"Error processing audio: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/api/synthesize")
async def synthesize_text(message: ChatMessage):
    """Handles HTTP POST requests for text-to-speech synthesis."""
    logger.debug(f"Received request for TTS with message: {message.message}")
    try:
        audio_file_path = text_to_speech(message.message)
        return FileResponse(audio_file_path, media_type='audio/mpeg', filename='output.mp3')
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Entry point for running with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
