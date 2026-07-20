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
    
    # Font fallback paths
    DEFAULT_FONT_PATH: str = "fonts/NotoSansMonoCJK-VF.ttf.ttc"
    FALLBACK_FONTS: list[str] = [
        "fonts/comic shanns 2.ttf",
        "fonts/NotoSans-Regular.ttf"
    ]

    # Playwright browser scraping
    # False = headed (thấy cửa sổ Chromium, dùng khi debug)
    # True  = headless (ẩn, dùng khi production)
    PLAYWRIGHT_HEADLESS: bool = False

    model_config = SettingsConfigDict(
        env_file="../.env",       # .env nằm ở root, server chạy từ BE/
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
