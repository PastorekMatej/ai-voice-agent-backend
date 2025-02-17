from fastapi import WebSocket
import openai

async def chat_with_ai(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = openai.ChatCompletion.create(
            model="gpt-4", messages=[{"role": "user", "content": data}]
        )
        await websocket.send_text(response["choices"][0]["message"]["content"])