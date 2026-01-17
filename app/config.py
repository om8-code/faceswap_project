import os
from pydantic import BaseModel

class Settings(BaseModel):
    DATA_DIR: str = os.getenv("DATA_DIR", "./app/data")
    BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8000")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_IMAGE_MODEL: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

    # retry behavior for 429
    GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
    GEMINI_RETRY_BASE_SECONDS: float = float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "2.0"))

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
