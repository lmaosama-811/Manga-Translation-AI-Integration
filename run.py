"""
Entry point: khởi động FastAPI server.

Cách chạy:
  - Từ terminal:  .venv\\Scripts\\python run.py
  - Hoặc dùng:    run.bat

Script này đảm bảo:
  1. CWD = BE/   → relative paths (../FE, ../library, fonts/) đúng
  2. sys.path có BE/  → "from app.xxx" resolve đúng kể cả uvicorn reload subprocess
  3. stdout/stderr UTF-8 trên Windows
"""

import sys
import io
import os
from pathlib import Path

# ── Xác định đường dẫn BE/ (tuyệt đối, không phụ thuộc CWD khi gọi) ──
BE_DIR = str((Path(__file__).parent / "BE").resolve())

# Thêm BE/ vào sys.path để "from app.xxx" luôn hoạt động
# (kể cả khi uvicorn reload spawn subprocess mới với CWD khác)
if BE_DIR not in sys.path:
    sys.path.insert(0, BE_DIR)

# Chuyển CWD sang BE/ để relative paths trong app code đúng
os.chdir(BE_DIR)

# Force stdout/stderr UTF-8 trên Windows
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
import uvicorn
from app.core.config import settings
from app.core.logging_config import setup_logging

if __name__ == "__main__":
    # Initialize logging before anything else fires
    setup_logging()

    # Windows: SelectorEventLoop (mặc định) không hỗ trợ subprocess.
    # Playwright cần spawn subprocess để launch Chromium → phải dùng ProactorEventLoop.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        reload_dirs=[BE_DIR],     # Chỉ watch BE/ khi reload
        log_config=None,          # Logging config riêng
    )
