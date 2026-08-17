import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "SWAYIN Predictor AI"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database (PostgreSQL with SQLite fallback)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///backend/swayin.db")
    
    # Redis Cache (With memory fallback)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_EXPIRE_SECONDS: int = 300
    
    # Gemma AI / LLM configuration
    GEMMA_API_KEY: str = os.getenv("GEMMA_API_KEY", "")
    GEMMA_MODEL_NAME: str = os.getenv("GEMMA_MODEL_NAME", "gemma-2-9b-it")
    
    # Market Data
    DEFAULT_SYMBOL: str = "^NSEI"
    DEFAULT_INTERVAL: str = "1d"
    DEFAULT_PERIOD: str = "5y"
    SUPPORTED_INTERVALS: List[str] = ["1m", "5m", "15m", "30m", "1h", "1d", "1w"]
    
    # Model Storage & Directory Config
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    SAVED_MODELS_DIR: str = os.path.join(BASE_DIR, "saved_models")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    
    class Config:
        case_sensitive = True

settings = Settings()

os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(settings.LOGS_DIR, exist_ok=True)
