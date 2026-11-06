# Architecture Documentation

## Overview

The AI Voice Agent Backend is a FastAPI-based REST API service designed to support French language teaching through voice interaction. It combines speech recognition, natural language processing, and text-to-speech synthesis.

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  (React Frontend, Mobile App, or any HTTP/WebSocket client)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP REST / WebSocket
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Application                          │
│                      (Port 5001)                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │   Health    │  │     Chat     │  │  Audio Pipeline  │      │
│  │  Endpoint   │  │   Endpoints  │  │   (STT / TTS)    │      │
│  └─────────────┘  └──────────────┘  └──────────────────┘      │
└────────────────────────────────────────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
        ┌───────────┐ ┌──────────┐ ┌─────────────┐
        │  OpenAI   │ │  Google  │ │   Google    │
        │    API    │ │  Speech  │ │     TTS     │
        │ GPT-3.5   │ │   STT    │ │             │
        └───────────┘ └──────────┘ └─────────────┘
```

## Core Components

### 1. FastAPI Application (`app/main.py`)

**Responsibilities:**
- HTTP request routing
- WebSocket connection management
- CORS configuration
- Request/response handling
- Error handling and logging

**Key Endpoints:**
- `GET /` - Service info
- `GET /health` - Health check
- `POST /api/chat` - Synchronous chat
- `POST /api/transcribe` - Audio → Text
- `POST /api/synthesize` - Text → Audio
- `WebSocket /chat` - Bidirectional real-time chat

### 2. Configuration Management (`app/config.py`)

**Responsibilities:**
- Environment variable loading
- Configuration validation
- Default value management

**Configuration Sources:**
1. Environment variables (highest priority)
2. `.env` file
3. Default values (fallback)

### 3. Service Layer (`app/services.py`)

**Responsibilities:**
- Google Cloud TTS/STT integration
- Voice settings management
- Audio processing utilities
- OpenAI chat helpers

**Key Functions:**
- `load_voice_settings()` - Load voice configuration
- `text_to_speech()` - Convert text to audio
- `speech_to_text()` - Convert audio to text
- `chat_completion()` - Get AI responses

### 4. SSML Builder (`app/ssml_builder.py`)

**Responsibilities:**
- Generate Speech Synthesis Markup Language
- Add natural pauses
- Control prosody (rate, pitch, volume)
- Add emphasis and expressiveness

**Features:**
- Fluent API for building SSML
- Support for breaks, emphasis, prosody
- Escaping of special characters

### 5. Performance Monitoring (`app/performance.py`)

**Responsibilities:**
- Track API response times
- Benchmark STT/TTS operations
- Log performance metrics

## Data Flow

### Chat Request Flow

```
1. Client sends POST /api/chat
   {
     "message": "Bonjour, comment ça va?"
   }
   
2. FastAPI receives request
   │
3. Extract and validate message
   │
4. Load voice settings (CECRL level, system prompt)
   │
5. Call OpenAI Chat Completions API
   │
6. Format response
   │
7. Return JSON response
   {
     "response": "Bonjour! Je vais très bien, merci..."
   }
```

### Transcription Flow

```
1. Client uploads audio (WebM format)
   │
2. FastAPI receives multipart file
   │
3. Save to temporary file
   │
4. Convert WebM → WAV using ffmpeg
   │
5. Call Google Cloud Speech-to-Text
   │
6. Process and format transcription
   │
7. Return transcription text
   │
8. Clean up temporary files
```

### Synthesis Flow

```
1. Client sends text for synthesis
   │
2. FastAPI receives request
   │
3. Build SSML with natural pauses
   │
4. Call Google Cloud Text-to-Speech
   │
5. Receive audio bytes (MP3)
   │
6. Save to temporary file
   │
7. Return audio file
   │
8. (⚠️ Currently: temp file not cleaned up)
```

## External Dependencies

### OpenAI API
- **Model**: gpt-3.5-turbo
- **Purpose**: Natural language chat responses
- **Authentication**: API key via environment variable
- **Rate Limits**: Depends on account tier

### Google Cloud Speech-to-Text
- **Language**: fr-FR (French)
- **Model**: default
- **Authentication**: Service account JSON
- **Input**: WAV, 16kHz, mono

### Google Cloud Text-to-Speech
- **Language**: fr-FR (French)
- **Voice**: Neural2 (varies by request)
- **Authentication**: Service account JSON
- **Output**: MP3 format

## Deployment Architectures

### Local Development

```
┌─────────────────┐
│  Developer PC   │
├─────────────────┤
│  • Python venv  │
│  • uvicorn      │
│  • Port 5001    │
│  • Hot reload   │
└─────────────────┘
```

### Docker Local

```
┌──────────────────────┐
│   Docker Container   │
├──────────────────────┤
│  • Python 3.11-slim  │
│  • ffmpeg            │
│  • Non-root user     │
│  • Health checks     │
│  • Port 5001         │
└──────────────────────┘
```

### Google Cloud Run (Production)

```
┌────────────────────────────────────┐
│       Google Cloud Platform         │
├────────────────────────────────────┤
│  ┌──────────────────────────────┐ │
│  │     Cloud Run Service        │ │
│  │  • Auto-scaling 0-100        │ │
│  │  • 1 vCPU, 1-2 GB RAM        │ │
│  │  • HTTPS endpoint            │ │
│  │  • Health monitoring         │ │
│  └──────────────────────────────┘ │
│                │                   │
│  ┌─────────────┴──────────────┐  │
│  │    Artifact Registry        │  │
│  │  Docker image storage       │  │
│  └─────────────────────────────┘  │
│                                    │
│  ┌─────────────────────────────┐  │
│  │    Secret Manager           │  │
│  │  • API keys                 │  │
│  │  • Service account creds    │  │
│  └─────────────────────────────┘  │
└────────────────────────────────────┘
```

## Security Considerations

### Current State

⚠️ **WARNING**: Current implementation has security gaps:

1. **No Authentication**: All endpoints publicly accessible
2. **CORS Misconfiguration**: `allow_origins=["*"]` with credentials
3. **Credentials in Code**: Hardcoded paths to service account files
4. **Debug Logging**: Potentially logs sensitive conversation data
5. **No Rate Limiting**: Vulnerable to abuse

### Recommended Security Measures

```
┌──────────────────────────────────────┐
│      Recommended Security Stack       │
├──────────────────────────────────────┤
│  1. API Key Authentication           │
│  2. JWT tokens for session mgmt      │
│  3. Rate limiting (e.g., slowapi)    │
│  4. Input validation (pydantic)      │
│  5. HTTPS only (TLS termination)     │
│  6. Secret Manager for credentials   │
│  7. CORS whitelist (specific origins)│
│  8. Request size limits              │
│  9. Audit logging                    │
│  10. Container scanning              │
└──────────────────────────────────────┘
```

## Scalability Considerations

### Current Bottlenecks

1. **Synchronous audio processing**: Blocks request thread
2. **Temporary file I/O**: Disk-bound operations
3. **No caching**: Repeated identical requests make full API calls
4. **Single-threaded OpenAI calls**: Sequential processing

### Scaling Strategies

**Horizontal Scaling (Cloud Run)**
- Auto-scaling based on request volume
- Each instance handles multiple concurrent requests
- Load balancer distributes traffic

**Performance Optimizations**
1. **Async audio processing**: Use background tasks
2. **In-memory buffering**: Avoid disk I/O where possible
3. **Response caching**: Cache common TTS phrases
4. **Connection pooling**: Reuse HTTP connections
5. **Streaming responses**: Stream audio instead of buffering

## Monitoring and Observability

### Current Capabilities

- Basic Python logging (DEBUG level)
- Container health checks
- Manual performance tracking (`app/performance.py`)

### Recommended Monitoring Stack

```
┌─────────────────────────────────────┐
│         Observability Stack          │
├─────────────────────────────────────┤
│  Metrics:                           │
│    • Prometheus/OpenTelemetry       │
│    • Request rate, latency          │
│    • Error rates                    │
│    • External API response times    │
│                                     │
│  Logging:                           │
│    • Structured JSON logs           │
│    • Google Cloud Logging           │
│    • Log aggregation                │
│                                     │
│  Tracing:                           │
│    • Distributed tracing            │
│    • Request path visualization     │
│                                     │
│  Alerting:                          │
│    • Error rate thresholds          │
│    • Latency SLOs                   │
│    • API quota warnings             │
└─────────────────────────────────────┘
```

## Future Architecture Evolution

### Phase 1: Stabilization (Current → v1.0)
- Fix security issues
- Add tests
- Implement proper error handling
- Clean up technical debt

### Phase 2: Feature Completeness (v1.0 → v2.0)
- Add conversation memory (database)
- Implement user sessions
- Track recurrent errors
- Generate study plans

### Phase 3: Scale (v2.0 → v3.0)
- Add Redis for session caching
- Implement background job queue
- Add real-time analytics
- Multi-language support

### Proposed v2.0 Architecture

```
┌────────────────────────────────────────────────────┐
│                 Load Balancer                       │
└──────────────┬─────────────────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────┐
    ▼                     ▼              ▼
┌─────────┐         ┌─────────┐     ┌─────────┐
│FastAPI 1│         │FastAPI 2│     │FastAPI N│
└────┬────┘         └────┬────┘     └────┬────┘
     │                   │               │
     └───────────┬───────┴───────────────┘
                 │
         ┌───────┴────────┐
         │                │
         ▼                ▼
    ┌────────┐      ┌──────────┐
    │ Redis  │      │PostgreSQL│
    │ Cache  │      │ Database │
    └────────┘      └──────────┘
```

## Development Guidelines

### Code Organization Principles

1. **Separation of Concerns**: Keep business logic, API handling, and external integrations separate
2. **Configuration Management**: All environment-specific values in config
3. **Error Handling**: Consistent error responses across all endpoints
4. **Logging**: Structured logging with appropriate levels
5. **Documentation**: Docstrings for all public functions

### Testing Strategy

```
┌────────────────────────────────────┐
│        Testing Pyramid              │
├────────────────────────────────────┤
│           /\                       │
│          /E2\      E2E Tests       │
│         /────\     (Few)           │
│        / Intg \    Integration     │
│       /────────\   (Some)          │
│      /  Unit    \  Unit Tests      │
│     /────────────\ (Many)          │
└────────────────────────────────────┘
```

**Unit Tests**: Test individual functions in isolation
- SSML builder
- Configuration loading
- Utility functions

**Integration Tests**: Test API endpoints with mocked external services
- Chat endpoint
- Transcription flow
- Synthesis flow

**E2E Tests**: Test complete flows with real services (staging)
- Full conversation cycle
- Audio transcription + response + synthesis

---

*For implementation details, see individual module documentation.*
*For deployment specifics, see README.md and Google Cloud documentation.*
