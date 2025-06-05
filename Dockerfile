# DOCKER IMPLEMENTATION: Enhanced Dockerfile for FastAPI backend with audio processing
FROM python:3.11-slim

WORKDIR /app

# DOCKER IMPLEMENTATION: Environment variables for container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/app/ai-voice-agent-451616-5ab9c7176a3d.json

# DOCKER IMPLEMENTATION: System dependencies including audio libraries for container
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# DOCKER IMPLEMENTATION: Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# DOCKER IMPLEMENTATION: Application code
COPY . .

# DOCKER IMPLEMENTATION: Create necessary directories for container runtime
RUN mkdir -p uploads config

# DOCKER IMPLEMENTATION: Security - non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5001

# DOCKER IMPLEMENTATION: Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# DOCKER IMPLEMENTATION: Use uvicorn directly for container startup
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5001"]