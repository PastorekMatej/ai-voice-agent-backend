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

## ☁️ Google Cloud Deployment

This section covers the complete deployment process for the AI Voice Agent backend on Google Cloud Platform using Cloud Run, Artifact Registry, and Google Cloud services integration.

### 🏗️ Google Cloud Configuration & Files

#### Required Files for Google Cloud Adaptation

The following files have been configured for Google Cloud deployment:

1. **`Dockerfile`** - Optimized for Cloud Run deployment
   - Multi-stage build for reduced image size
   - FastAPI production server (Uvicorn) configuration
   - Google Cloud services integration
   - Health checks for Cloud Run monitoring
   - Proper file permissions and user configuration

2. **`requirements.txt`** - Python dependencies
   - FastAPI and Uvicorn for the web server
   - Google Cloud client libraries (Speech, Text-to-Speech)
   - OpenAI Python client
   - Audio processing libraries (pydub, ffmpeg)
   - CORS and multipart handling

3. **`ai-voice-agent-451616-5ab9c7176a3d.json`** - Google Cloud service account credentials
   - Service account with necessary permissions
   - Speech-to-Text API access
   - Text-to-Speech API access
   - Cloud Storage access (if needed)

4. **`.dockerignore`** - Build optimization
   - Excludes development files and cache
   - Reduces build context size
   - Improves build performance

#### Environment Variables Configuration

```bash
# Required for Google Cloud deployment
OPENAI_API_KEY=your-openai-api-key
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
ENVIRONMENT=production
PORT=8080
HOST=0.0.0.0
FRONTEND_URL=https://ai-voice-frontend-446760904661.us-central1.run.app
```

### 🛠️ Prerequisites

Before deploying to Google Cloud, ensure you have:

```bash
# 1. Google Cloud CLI installed and authenticated
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Enable required APIs
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable speech.googleapis.com
gcloud services enable texttospeech.googleapis.com

# 3. Create or verify Artifact Registry repository
gcloud artifacts repositories create ai-voice-agent \
    --repository-format=docker \
    --location=us-central1 \
    --description="AI Voice Agent containers"

# 4. Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

# 5. Create service account for backend services
gcloud iam service-accounts create ai-voice-backend-sa \
    --display-name="AI Voice Backend Service Account"

# 6. Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:ai-voice-backend-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/speech.client"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:ai-voice-backend-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/texttospeech.client"
```

### 🚀 Key Google Cloud Deployment Commands

#### Method 1: Manual Docker Build & Deploy

```bash
# 1. Build the Docker image
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ai-voice-agent/backend:latest .

# 2. Push to Artifact Registry
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ai-voice-agent/backend:latest

# 3. Deploy to Cloud Run
gcloud run deploy ai-voice-backend \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ai-voice-agent/backend:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 100 \
    --min-instances 0 \
    --set-env-vars="OPENAI_API_KEY=your-openai-key,ENVIRONMENT=production,GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json"
```

#### Method 2: Cloud Build (Recommended)

Create `cloudbuild.yaml` in the backend directory:

```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: 
      - 'build'
      - '-t'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/ai-voice-agent/backend:$COMMIT_SHA'
      - '.'
  
  # Push the container image to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'us-central1-docker.pkg.dev/$PROJECT_ID/ai-voice-agent/backend:$COMMIT_SHA']
  
  # Deploy container image to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'ai-voice-backend'
      - '--image=us-central1-docker.pkg.dev/$PROJECT_ID/ai-voice-agent/backend:$COMMIT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--port=8080'
      - '--memory=1Gi'
      - '--cpu=1'
      - '--timeout=300'
      - '--max-instances=100'
      - '--set-env-vars=ENVIRONMENT=production,GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json'

images:
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/ai-voice-agent/backend:$COMMIT_SHA'

options:
  logging: CLOUD_LOGGING_ONLY
```

Deploy with Cloud Build:
```bash
gcloud builds submit --config cloudbuild.yaml .
```

#### Method 3: Secret Manager Integration (Recommended for Production)

```bash
# Store OpenAI API key in Secret Manager
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-

# Grant access to the service account
gcloud secrets add-iam-policy-binding openai-api-key \
    --member="serviceAccount:ai-voice-backend-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Deploy with secret environment variable
gcloud run deploy ai-voice-backend \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ai-voice-agent/backend:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --service-account ai-voice-backend-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --set-secrets="OPENAI_API_KEY=openai-api-key:latest" \
    --set-env-vars="ENVIRONMENT=production,GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json"
```

### 🔧 Advanced Cloud Run Configuration

#### Update Service Configuration
```bash
# Update environment variables
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --set-env-vars ENVIRONMENT=production,DEBUG=false

# Update resource allocation for heavy AI workloads
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 600

# Configure concurrency for audio processing
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --concurrency 10 \
    --max-instances 50
```

#### Custom Domain and SSL
```bash
# Map custom domain
gcloud run domain-mappings create \
    --service ai-voice-backend \
    --domain api.your-domain.com \
    --region us-central1

# Verify SSL certificate
gcloud run domain-mappings describe \
    --domain api.your-domain.com \
    --region us-central1
```

### 📊 Monitoring & Management Commands

#### Service Management
```bash
# Get service details and URL
gcloud run services describe ai-voice-backend --region us-central1

# Get backend service URL
BACKEND_URL=$(gcloud run services describe ai-voice-backend \
    --region us-central1 \
    --format="value(status.url)")
echo $BACKEND_URL
```

#### Health Monitoring
```bash
# Test health endpoint
curl $BACKEND_URL/health

# Test API endpoints
curl $BACKEND_URL/

# Test chat endpoint
curl -X POST $BACKEND_URL/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, test message"}'
```

#### Logs & Debugging
```bash
# View service logs
gcloud run services logs read ai-voice-backend --region us-central1

# Follow logs in real-time
gcloud run services logs tail ai-voice-backend --region us-central1

# Filter logs by severity
gcloud run services logs read ai-voice-backend \
    --region us-central1 \
    --log-filter="severity>=ERROR"

# View specific time range
gcloud run services logs read ai-voice-backend \
    --region us-central1 \
    --log-filter="timestamp>=\"2024-01-01T00:00:00Z\""
```

### 🔐 Security & IAM Configuration

#### Service Account Best Practices
```bash
# Create dedicated service account
gcloud iam service-accounts create ai-voice-backend-prod \
    --display-name="AI Voice Backend Production"

# Grant minimal required permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:ai-voice-backend-prod@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/speech.client"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:ai-voice-backend-prod@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/texttospeech.client"

# Update service to use new service account
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --service-account ai-voice-backend-prod@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

#### Network Security
```bash
# Restrict ingress (if needed)
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --ingress internal-and-cloud-load-balancing

# Configure VPC connector for internal services
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --vpc-connector YOUR_VPC_CONNECTOR \
    --vpc-egress private-ranges-only
```

### 🚨 Troubleshooting Google Cloud Deployment

#### Common Backend Issues & Solutions

**Audio Processing Failures:**
```bash
# Check if ffmpeg is properly installed in container
gcloud run services logs read ai-voice-backend --region us-central1 | grep ffmpeg

# Test audio endpoint specifically
curl -X POST $BACKEND_URL/api/transcribe \
    -F "audio=@test_audio.wav"
```

**Google Cloud API Issues:**
```bash
# Verify service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID \
    --filter="bindings.members:ai-voice-backend-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com"

# Test Speech-to-Text API access
gcloud auth activate-service-account --key-file=credentials.json
gcloud ml speech recognize test_audio.wav --language-code=en-US
```

**OpenAI API Connection Issues:**
```bash
# Check if API key is properly set
gcloud run services describe ai-voice-backend \
    --region us-central1 \
    --format="value(spec.template.spec.template.spec.containers[0].env[].name)"

# Test OpenAI connectivity from Cloud Shell
curl -X POST $BACKEND_URL/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Test OpenAI connection"}'
```

**Memory and Timeout Issues:**
```bash
# Increase memory and timeout for audio processing
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --memory 2Gi \
    --timeout 900

# Monitor resource usage
gcloud run services describe ai-voice-backend \
    --region us-central1 \
    --format="table(spec.template.spec.template.spec.containers[0].resources)"
```

### 💰 Cost Optimization

```bash
# Set appropriate minimum instances
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --min-instances 0  # For cost optimization

# Configure request timeout to prevent long-running requests
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --timeout 300

# Monitor and set concurrency limits
gcloud run services update ai-voice-backend \
    --region us-central1 \
    --concurrency 10
```

### 🔄 Rollback & Version Management

```bash
# List all revisions
gcloud run revisions list --service ai-voice-backend --region us-central1

# Rollback to previous revision
gcloud run services update-traffic ai-voice-backend \
    --to-revisions ai-voice-backend-00001=100 \
    --region us-central1

# Gradual rollout (canary deployment)
gcloud run services update-traffic ai-voice-backend \
    --to-revisions ai-voice-backend-00001=90,ai-voice-backend-00002=10 \
    --region us-central1
```

### 🌐 Production Backend Deployment Checklist

- [ ] Google Cloud APIs enabled (Speech-to-Text, Text-to-Speech)
- [ ] Service account created with proper permissions
- [ ] OpenAI API key stored securely (Secret Manager)
- [ ] Google Cloud credentials configured
- [ ] Dockerfile optimized for production
- [ ] Health checks implemented and working
- [ ] CORS configured for frontend URL
- [ ] Resource limits appropriate for workload
- [ ] Monitoring and logging configured
- [ ] SSL/TLS certificates provisioned
- [ ] Error handling and retry logic implemented
- [ ] Audio processing pipeline tested
- [ ] Load testing completed

### 🔗 Integration with Frontend

After backend deployment, update your frontend configuration:

```bash
# Get backend URL
BACKEND_URL=$(gcloud run services describe ai-voice-backend \
    --region us-central1 \
    --format="value(status.url)")

# Update frontend with backend URL
echo "Backend URL for frontend: $BACKEND_URL"
```

Use this URL to update the frontend's `REACT_APP_API_URL` environment variable.

### 📚 Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text/docs)
- [Google Cloud Text-to-Speech API](https://cloud.google.com/text-to-speech/docs)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Google Cloud Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Container Security Best Practices](https://cloud.google.com/run/docs/securing/services)

