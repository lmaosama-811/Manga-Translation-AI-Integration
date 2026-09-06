"""
Module: app.core.config
Description: Defines system configurations and environment variables using Pydantic Settings.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API endpoints and keys
    GEMINI_API_KEYS: list[str] = [""]          # Free tier keys — Async + Sync mode
    GEMINI_PRO_API_KEYS: list[str] = [""]      # Pro/Paid tier keys — Rich Mode only
    OLLAMA_API_URL: str = "http://localhost:11434/api/chat"

    # Redis / Celery broker
    REDIS_URL: str = "redis://localhost:6379/0"  # Upstash URL khi deploy cloud

    # PostgreSQL database
    DATABASE_URL: str = ""
    
    # Fallback engine configuration
    GLOBAL_RETRY_MAX: int = 1           # Số lần retry toàn cục khi tất cả key+model đều fail
    GLOBAL_RETRY_BACKOFF_SEC: int = 30  # Thời gian chờ (giây) trước khi global retry
    
    # Server runtime configurations
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Image constraints
    MAX_IMAGE_SIZE: int = 2048

    # RT-DETR-v2 PyTorch FP32 text block detector
    DETECTOR_MODEL_PATH: str = "ogkalu/comic-text-and-bubble-detector"
    DETECTOR_CONF_THRESH: float = 0.30  # Ngưỡng confidence score (0.0 - 1.0)

    # Font fallback paths
    DEFAULT_FONT_PATH: str =  "fonts/VNF-Comic Sans.ttf"
    FALLBACK_FONTS: list[str] = [
        "fonts/000 CCDaveGibbons Regular.ttf",
        "fonts/000 CCJoeKubert Regular.ttf",
        "fonts/000 CCMildMannered Regular.ttf",
        "fonts/000BlahBlahUCiCiel-Regular.ttf",
        "fonts/000 Comic Pro JY.ttf",
        "fonts/000 CCVictorySpeech Regular.ttf",
        "fonts/000 Collect Em All BB Regular.ttf"
    ]

    # Playwright browser scraping
    # False = headed (thấy cửa sổ Chromium, dùng khi debug)
    # True  = headless (ẩn, dùng khi production)
    PLAYWRIGHT_HEADLESS: bool = False

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),       # Tìm .env ở thư mục hiện tại hoặc thư mục cha
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
