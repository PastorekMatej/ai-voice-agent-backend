# AI Voice Agent Backend

[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)]()

## 🔔 Important Notice About Branches

**⚠️ The `main` branch contains only a minimal stub implementation.**

The actual working application exists on:
- **`develop-elevenlabs`** (recommended) - Full application with comprehensive Google Cloud deployment documentation
- **`develop`** - Core application functionality

This README documents the full application architecture found on the development branches.

---

## 📋 Overview

AI teaching assistant backend designed for French language instruction. The system provides:
- **Speech-to-Text** transcription (French, using Google Cloud Speech API)
- **AI-powered chat** responses (OpenAI GPT-3.5-turbo)
- **Text-to-Speech** synthesis with SSML support (French Neural2 voices via Google Cloud)
- **WebSocket** support for real-time communication

### Intended Use Case
French language teaching assistant capable of analyzing speech and providing feedback. The roadmap includes memory-based analysis of recurrent errors, study plan generation, and progress tracking (not yet implemented).

---

## 🏗️ Architecture

```
┌─────────────────────┐
│   Frontend Client   │
│   (Port 3000)       │
└──────────┬──────────┘
           │ HTTP/WebSocket
           ▼
┌─────────────────────┐
│  FastAPI Backend    │
│   (Port 5001)       │
├─────────────────────┤
│ • /api/chat         │◄──► OpenAI GPT-3.5
│ • /api/transcribe   │◄──► Google Speech-to-Text
│ • /api/synthesize   │◄──► Google Text-to-Speech
│ • /chat (WebSocket) │
└─────────────────────┘
```

### Main Components

| Module | Purpose | Lines of Code |
|--------|---------|---------------|
| `app/main.py` | FastAPI application, API endpoints, WebSocket handler | ~440 |
| `app/config.py` | Configuration management (env vars) | ~30 |
| `app/services.py` | Voice settings, TTS/STT helpers | ~150 |
| `app/ssml_builder.py` | SSML generation for expressive speech | ~100 |
| `app/performance.py` | Performance metrics and benchmarking | ~80 |

---

## 🛠️ Technology Stack

### Core Dependencies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Web Framework** | FastAPI | 0.115.8 | REST API and WebSocket server |
| **ASGI Server** | Uvicorn | 0.24.0 | Production-ready async server |
| **AI/LLM** | OpenAI | 1.63.0 | Chat completions (GPT-3.5-turbo) |
| **Speech-to-Text** | Google Cloud Speech | 2.21.0 | French audio transcription |
| **Text-to-Speech** | Google Cloud TTS | 2.16.3 | French voice synthesis |
| **Audio Processing** | ffmpeg | - (system) | Audio format conversion (WebM→WAV) |
| **Audio Library** | pydub | 0.25.1 | Audio manipulation |

### System Requirements

- **Python**: 3.11+
- **ffmpeg**: Required for audio transcription
- **PortAudio**: Required for audio device access (development)
- **Docker**: Optional, for containerized deployment

---

## 🚀 Installation & Setup

### Prerequisites

1. **Python 3.11 or higher**
2. **ffmpeg** installed and available on PATH:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

3. **API Keys & Credentials**:
   - OpenAI API key ([get one here](https://platform.openai.com/api-keys))
   - Google Cloud service account JSON with Speech-to-Text and Text-to-Speech APIs enabled

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Required for STT/TTS endpoints
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json

# Optional (defaults shown)
HOST=0.0.0.0
PORT=5001
FRONTEND_URL=http://localhost:3000
ENVIRONMENT=development
```

### Local Development Setup

```bash
# 1. Clone and checkout the development branch
git clone https://github.com/PastorekMatej/ai-voice-agent-backend.git
cd ai-voice-agent-backend
git checkout develop-elevenlabs

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set environment variables
export OPENAI_API_KEY="your_key_here"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# 6. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload
```

The API will be available at `http://localhost:5001`

### Docker Deployment

```bash
# Build the image
docker build -t ai-voice-agent-backend .

# Run the container
docker run -p 5001:5001 \
  -e OPENAI_API_KEY="your_key_here" \
  -e ENVIRONMENT="production" \
  -v /path/to/credentials.json:/app/ai-voice-agent-451616-5ab9c7176a3d.json \
  --name ai-voice-backend \
  ai-voice-agent-backend
```

**Note**: The Dockerfile expects credentials at a specific path. Update the volume mount accordingly.

---

## 📡 API Endpoints

### Health & Info

- **GET** `/health` - Health check endpoint (returns 200 OK when service is running)
- **GET** `/` - Service information

### Chat Endpoints

- **POST** `/api/chat`
  ```json
  {
    "message": "Bonjour, comment ça va?"
  }
  ```
  Returns AI-generated response adapted for French language teaching.

- **WebSocket** `/chat` - Real-time bidirectional chat over WebSocket

### Audio Processing

- **POST** `/api/transcribe`
  - Content-Type: `multipart/form-data`
  - Field: `audio` (WebM format)
  - Returns: French transcription text

- **POST** `/api/synthesize`
  ```json
  {
    "message": "Texte à synthétiser"
  }
  ```
  Returns: MP3 audio file with French speech

---

## 🧪 Testing

### Manual Testing

```bash
# Test health endpoint
curl http://localhost:5001/health

# Test chat endpoint
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bonjour!"}'

# Test transcription (with an audio file)
curl -X POST http://localhost:5001/api/transcribe \
  -F "audio=@test_audio.wav"
```

### Current Test Status

**⚠️ No automated tests are currently implemented.**

- No test files in the repository
- No CI/CD pipelines configured
- Testing is currently manual only

---

## ☁️ Google Cloud Deployment

The `develop-elevenlabs` branch includes comprehensive Google Cloud Run deployment documentation. Key steps:

1. **Enable required APIs**:
   - Cloud Run
   - Artifact Registry
   - Speech-to-Text
   - Text-to-Speech

2. **Build and push to Artifact Registry**:
   ```bash
   docker build -t us-central1-docker.pkg.dev/PROJECT_ID/ai-voice-agent/backend:latest .
   docker push us-central1-docker.pkg.dev/PROJECT_ID/ai-voice-agent/backend:latest
   ```

3. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy ai-voice-backend \
     --image us-central1-docker.pkg.dev/PROJECT_ID/ai-voice-agent/backend:latest \
     --platform managed \
     --region us-central1 \
     --port 5001 \
     --memory 1Gi \
     --set-env-vars="OPENAI_API_KEY=...,ENVIRONMENT=production"
   ```

**Note**: See the README on `develop-elevenlabs` for complete deployment documentation including Secret Manager integration, monitoring, and troubleshooting.

---

## ⚠️ Known Issues & Risks

### Critical Issues

1. **Branch Divergence**
   - `main` branch is a minimal stub (~50 lines)
   - Production code exists only on `develop` and `develop-elevenlabs`
   - **Recommendation**: Merge development branches into `main` or update default branch

2. **Committed node_modules Directory**
   - ~1,900 JavaScript files (28MB) committed under `app/node_modules/`
   - Contains Google Cloud Speech SDK for JavaScript (not used by Python backend)
   - No `package.json` at root
   - **Recommendation**: Remove from repository and add to `.gitignore`

3. **Missing Configuration File**
   - `config/voice_settings.json` referenced in code but absent from all branches
   - Code falls back to minimal defaults
   - **Recommendation**: Add the missing configuration file or remove references

4. **CORS Security Misconfiguration**
   - `allow_origins=["*"]` with `allow_credentials=True`
   - Invalid configuration for browsers
   - **Recommendation**: Use specific allowed origins list

5. **Hardcoded Credentials Paths**
   - Dockerfile expects specific credential filename
   - Paths include personal Windows directories (`C:\Users\pasto\...`)
   - **Recommendation**: Use environment-based paths

### Medium Priority Issues

6. **Port Configuration Mismatch**
   - Dockerfile uses port `5001`
   - Google Cloud documentation examples use `8080`
   - **Recommendation**: Make port configurable via `$PORT` environment variable

7. **API Version Inconsistency**
   - Mix of legacy OpenAI API (`openai.Completion.create`) and new client (`OpenAI()`)
   - `main` branch uses deprecated models (`text-davinci-003`)
   - **Recommendation**: Standardize on new OpenAI client API

8. **Startup API Probe**
   - Application makes live OpenAI API call at startup for validation
   - Costs tokens and slows container initialization
   - Fails boot without network access
   - **Recommendation**: Move validation to first request or make optional

9. **Temporary File Leaks**
   - `/api/synthesize` creates `/tmp/output_*.mp3` files without cleanup
   - **Recommendation**: Add cleanup logic or use tempfile context managers

10. **Debug Logging in Production**
    - `logging.DEBUG` level always enabled
    - May log sensitive user conversations
    - **Recommendation**: Make log level configurable via environment

11. **No Authentication**
    - All endpoints publicly accessible
    - README suggests `--allow-unauthenticated` for Cloud Run
    - **Recommendation**: Add authentication for production use

12. **Function Shadowing**
    - Functions imported from `services.py` then redefined in `main.py`
    - Can cause confusion about which implementation is used
    - **Recommendation**: Consolidate to single implementation

### Missing Features (Documented in Roadmap)

- Memory/conversation history
- Recurrent error analysis
- Study plan generation
- Progress tracking
- ElevenLabs TTS integration (despite branch name)

---

## 📊 CI/CD Status

**Current Status**: No CI/CD pipelines configured

- No `.github/workflows/` directory
- No automated testing
- No automated deployment
- Manual deployment to Google Cloud Run

**Recommendations**:
- Add GitHub Actions workflow for:
  - Linting (pylint, flake8, black)
  - Unit tests (pytest)
  - Docker build validation
  - Automated deployment to staging environment

---

## 🗂️ Repository Structure

### Current `main` Branch (stub only)
```
/workspace
├── app/
│   ├── main.py           # Basic stub (~26 lines)
│   ├── config.py         # Minimal config
│   ├── services.py       # Legacy stubs
│   ├── websocket.py      # Unused
│   └── node_modules/     # ⚠️ Should not be committed
├── requirements.txt      # Unpinned dependencies
└── start.sh             # Basic start script
```

### `develop-elevenlabs` Branch (full application)
```
/workspace
├── app/
│   ├── main.py              # Full FastAPI app (~440 lines)
│   ├── config.py            # Complete configuration
│   ├── services.py          # TTS/STT/chat services
│   ├── ssml_builder.py      # SSML generation
│   ├── performance.py       # Metrics/benchmarking
│   └── websocket.py         # WebSocket helpers (unused)
├── Dockerfile              # Production-ready container
├── .dockerignore          # Build optimization
├── .env.example           # Environment template
├── .gitignore             # Git ignore rules
├── requirements.txt       # Pinned dependencies
├── README.md             # Comprehensive documentation
└── start.sh              # Start script
```

---

## 🔄 Version History

Tags track evolution on development branches:
- `v0.1.0` - Initial version
- `v0.2.0` - French voice integration
- `v0.3.0` - Benchmarking additions
- `v0.4.0` - Services centralization
- `v0.5.0` - Performance improvements
- `v0.6.0` - Latest tagged version

---

## 📝 Development Recommendations

### Immediate Actions
1. ✅ **Merge or rebase** `develop-elevenlabs` into `main`
2. ✅ **Remove** `app/node_modules/` from git history
3. ✅ **Add** missing `config/voice_settings.json`
4. ✅ **Fix** CORS configuration
5. ✅ **Create** `.env.example` on `main`
6. ✅ **Update** `.gitignore` to exclude `node_modules/`

### Short-term Improvements
- Add unit tests (pytest)
- Add integration tests for API endpoints
- Set up GitHub Actions CI/CD
- Add proper authentication
- Implement request rate limiting
- Add API documentation (Swagger/OpenAPI)
- Configure proper logging levels

### Long-term Features (from Roadmap)
- Implement conversation memory
- Add recurrent error tracking
- Develop study plan generation
- Create progress tracking dashboard
- Integrate ElevenLabs TTS (if desired)

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text/docs)
- [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Docker Documentation](https://docs.docker.com/)

---

## 📄 License

*No license file currently in repository*

## 🤝 Contributing

*No contributing guidelines currently in repository*

---

## 📞 Support

For issues or questions, please open an issue on GitHub.

**Repository**: [PastorekMatej/ai-voice-agent-backend](https://github.com/PastorekMatej/ai-voice-agent-backend)

---

*Documentation created through automated repository analysis. Last updated: November 2026*
