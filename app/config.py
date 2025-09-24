# app/config.py

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- Database Configuration ---
    # Format: postgresql+asyncpg://user:password@host:port/dbname
    DATABASE_URL: str = "postgresql+asyncpg://postgres:admin@localhost:5432/employee"

    # --- Model Configuration ---
    MODEL_PATH: str = r"model/buffalo_l/glintr100.onnx"

    # --- Directory Configuration ---
    IMAGE_UPLOAD_FOLDER: str = "uploads"
    DEBUG_SAVE_DIR: str = "debug_uploads"
    
    # --- Recognition Threshold ---
    RECOGNITION_THRESHOLD: float = 0.35

    class Config:
        # If you use a .env file, settings will be loaded from it
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# --- Create directories on startup ---
os.makedirs(settings.IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(settings.DEBUG_SAVE_DIR, exist_ok=True)