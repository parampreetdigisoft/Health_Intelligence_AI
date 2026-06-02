"""
Enhanced Configuration with Multi-LLM Provider Support
"""
import os
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM Providers"""
    OPENAI = "openai"
    OPENROUTER = "openrouter"

class Settings:
    # ---------------------------
    # API Configuration
    # ---------------------------
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # ---------------------------
    # .NET API Integration
    # ---------------------------
    DOTNET_API_URL: str = os.getenv("DOTNET_API_URL", "https://api.veridianurbansystems.com")
    
    Application_Auth_API_KEY: str= os.getenv("Application_Auth_API_KEY", "v1-abac4b8a0947535005d4595b2c05fce0b3ae9ab2872451d3d259451f36e03bd2")

    # ---------------------------
    # Database Configuration
    # ---------------------------
    DB_SERVER: str = os.getenv("DB_SERVER", "DESKTOP-I0TTFPS")
    DB_NAME: str = os.getenv("DB_NAME", "AssessmentDB_new")
    DB_USE_WINDOWS_AUTH: bool = os.getenv("DB_USE_WINDOWS_AUTH", "True").lower() == "true"
    DB_USERNAME: str = os.getenv("DB_USERNAME", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    # ---------------------------
    # LLM Provider Configuration
    # ---------------------------
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    
    # OpenRouter Configuration (uses OpenAI-compatible API)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
 

    # ---------------------------
    # Data Analysis Configuration
    # ---------------------------
    MAX_RECORDS_FOR_ANALYSIS: int = int(os.getenv("MAX_RECORDS_FOR_ANALYSIS", "1000"))
    SAMPLE_SIZE: int = int(os.getenv("SAMPLE_SIZE", "100"))
    USE_SAMPLING: bool = os.getenv("USE_SAMPLING", "True").lower() == "true"
    
    # ---------------------------
    # Processing Settings
    # ---------------------------
    MAX_SUMMARY_LENGTH: int = 150
    MIN_SUMMARY_LENGTH: int = 50
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 5000
    
    # ---------------------------
    # Scoring
    # ---------------------------
    ANOMALY_THRESHOLD: int = 2
    
    # ---------------------------
    # General Paths
    # ---------------------------
    BASE_DIR: Path = Path(__file__).parent.parent
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

