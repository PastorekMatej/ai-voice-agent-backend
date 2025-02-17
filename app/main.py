# backend/main.py

import openai
import os
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Voice Agent!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = openai.Completion.create(
            model="text-davinci-003", prompt=data, max_tokens=50
        )
        await websocket.send_text(response.choices[0].text.strip())