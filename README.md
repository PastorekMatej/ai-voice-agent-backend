# AI Voice Agent Backend

## Status
🚧 Work in progress – baseline code pushed, README and documentation coming soon.

AI agent used as a teaching assistant for teachers of French language. 
The assistant could be deployed for communication and more accurate identification of recurent errors. 

## Roadmap 
AI agent is capable to analyse the speech and provide feedback.
Integrated memory allows agent to analyse recurent errors. Agent is able to propose study plan and track improvement.

## 🐳 Docker Implementation

### Overview
This backend is fully containerized using Docker for consistent deployment across different environments. The Docker implementation includes:

- **FastAPI backend** with all API endpoints
- **Google Cloud services** integration (Speech-to-Text, Text-to-Speech)
- **OpenAI GPT integration** for intelligent responses
- **Audio processing** with ffmpeg for WebM to WAV conversion
- **Health checks** for container monitoring
- **Environment-based configuration** for development and production

### 🚀 Quick Start with Docker

#### Prerequisites
- Docker installed and running
- OpenAI API key
- Google Cloud credentials JSON file

#### Build and Run
```bash
# Build the Docker image
docker build -t ai-voice-agent-backend .

# Run the container
docker run -p 5001:5001 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e ENVIRONMENT="development" \
  --name ai-voice-backend \
  ai-voice-agent-backend
```

#### Environment Variables
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `ENVIRONMENT` - Set to "development" or "production"
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 5001)
- `FRONTEND_URL` - Frontend URL for CORS (default: http://localhost:3000)

### 📡 API Endpoints

- `GET /health` - Health check endpoint
- `GET /` - Service information
- `POST /api/chat` - AI chat conversation
- `POST /api/transcribe` - Speech-to-text conversion
- `POST /api/synthesize` - Text-to-speech synthesis
- `WebSocket /chat` - Real-time chat communication

### 🔧 Development

#### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload
```

#### Testing
```bash
# Test health endpoint
curl http://localhost:5001/health

# Test chat endpoint
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

