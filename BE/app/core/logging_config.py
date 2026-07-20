"""
Module: app.core.logging_config
Description: Centralized logging setup for the entire application.

Features:
- Console handler with colored level labels (via a simple formatter)
- Rotating file handler → logs/app.log (max 10 MB per file, 5 backups)
- Separate rotating file for errors only → logs/error.log
- Configurable log level via LOG_LEVEL env variable (default: INFO)
- All third-party library loggers are quieted to WARNING to reduce noise

Usage:
    from app.core.logging_config import setup_logging
    setup_logging()   # Call once at process startup (run.py)
"""

import logging
import logging.handlers
import os
import sys

# ---------------------------------------------------------------------------
# Log directory — created automatically if it does not exist
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

APP_LOG_PATH   = os.path.join(LOG_DIR, "app.log")
ERROR_LOG_PATH = os.path.join(LOG_DIR, "error.log")


# ---------------------------------------------------------------------------
# Custom formatter — adds color to console output (ANSI, works on Windows 10+)
# ---------------------------------------------------------------------------
LEVEL_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
}
RESET = "\033[0m"


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes around the level name."""

    BASE_FMT = "%(asctime)s | {color}%(levelname)-8s{reset} | %(name)s | %(message)s"
    DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS.get(record.levelname, "")
        fmt = self.BASE_FMT.format(color=color, reset=RESET)
        formatter = logging.Formatter(fmt, datefmt=self.DATE_FMT)
        return formatter.format(record)


class PlainFormatter(logging.Formatter):
    """Plain formatter for file handlers (no ANSI codes)."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


# ---------------------------------------------------------------------------
# Public setup function
# ---------------------------------------------------------------------------

def setup_logging(level: str | None = None) -> None:
    """
    Configure the root logger and all application-level loggers.

    Args:
        level: Override log level string (e.g. "DEBUG", "INFO").
               Falls back to the LOG_LEVEL env variable, then "INFO".
    """
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # ── Root logger ──────────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid adding duplicate handlers if setup_logging() is called more than once
    if root_logger.handlers:
        root_logger.handlers.clear()

    # ── Console handler (colored) ─────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredConsoleFormatter())
    root_logger.addHandler(console_handler)

    # ── Rotating file handler — all levels → logs/app.log ─────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        APP_LOG_PATH,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(PlainFormatter())
    root_logger.addHandler(file_handler)

    # ── Error-only rotating file handler → logs/error.log ─────────────────────
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG_PATH,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(PlainFormatter())
    root_logger.addHandler(error_handler)

    # ── Quiet noisy third-party loggers ───────────────────────────────────────
    _noisy_libs = [
        "httpx",
        "httpcore",
        "uvicorn.access",
        "transformers",
        "PIL",
        "torch",
        "timm",
    ]
    for lib in _noisy_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # First log line so we can confirm setup worked
    logging.getLogger(__name__).info(
        f"Logging initialized — level={log_level_str} | "
        f"app.log → {APP_LOG_PATH} | error.log → {ERROR_LOG_PATH}"
    )
