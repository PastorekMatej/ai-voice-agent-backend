# DOCKER IMPLEMENTATION: Enhanced config for containerized environment
import os
from dotenv import load_dotenv

# DOCKER IMPLEMENTATION: Load .env file only if it exists (for local dev)
if os.path.exists('.env'):
    load_dotenv()

# DOCKER IMPLEMENTATION: Configuration with fallbacks for container deployment
class Config:
    # DOCKER IMPLEMENTATION: API keys from environment variables (set in container runtime)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # DOCKER IMPLEMENTATION: Server configuration for containerized deployment
    HOST = os.getenv("HOST", "0.0.0.0")  # Bind to all interfaces in container
    PORT = int(os.getenv("PORT", 5001))
    
    # DOCKER IMPLEMENTATION: CORS settings for frontend-backend communication in containers
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # DOCKER IMPLEMENTATION: Environment detection for container deployment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# DOCKER IMPLEMENTATION: Export config instance for use in main.py
config = Config()