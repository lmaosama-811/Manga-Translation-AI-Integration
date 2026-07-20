"""
Module: app.main
Description: Initializes the FastAPI application, registers API routes, and serves the static HTML playground interface.

Path conventions (CWD = BE/ khi chạy từ run.py ở root):
  ../FE/static/   — HTML pages (playground, reader)
  ../library/     — Ảnh manga đã dịch, mounted tại /library
"""

import sys
import io
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.logging_config import setup_logging
from app.core.config import settings
# Initialize logging (ensures logging works in uvicorn child reload processes)
setup_logging()

import logging
logger = logging.getLogger(__name__)

from app.api.routes.translate import router as translate_router
from app.api.routes.scrape import router as scrape_router

# Force stdout/stderr to use UTF-8 encoding on Windows to prevent UnicodeEncodeError for Vietnamese characters
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: tạo DB tables nếu chưa tồn tại (idempotent — safe to call every time)
    if settings.DATABASE_URL:
        try:
            from app.services.db_service import init_db
            await init_db()
        except Exception as e:
            logger.warning(f"[Startup] DB init warning (non-fatal): {e}")
    yield
    # Shutdown: close Playwright browser nếu đã được khởi tạo
    try:
        from app.services.browser_scraper import close_browser
        await close_browser()
    except Exception:
        pass


app = FastAPI(
    title="Manga VLM Translator Production API",
    description="Production-ready API for manga OCR, translation, and high-fidelity typesetting using VLM & LaMa.",
    version="2.0",
    lifespan=lifespan,
)

# CORS — cho phép Chrome Extension (chrome-extension://*) và localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "chrome-extension://*",  # Chrome Extension origin
        "*",  # Cho test; giới hạn lại khi production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HTML routes ─────────────────────────────────────────────────────────────
# Serve playground interface directly at root URL '/'
@app.get("/", response_class=FileResponse)
async def serve_playground():
    return FileResponse("../FE/static/playground.html")

# Manga reader page — poll job status + hiển thị ảnh dịch progressive
@app.get("/reader", response_class=FileResponse)
async def serve_reader():
    return FileResponse("../FE/static/reader.html")

# ── API routes ───────────────────────────────────────────────────────────────
app.include_router(translate_router)
app.include_router(scrape_router)

# ── Static file serving ──────────────────────────────────────────────────────

# Serve library (ảnh manga đã dịch) — cấu trúc: library/{name} [{uuid8}]/{chapter}/
LIBRARY_DIR = Path("../library")
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/library", StaticFiles(directory=str(LIBRARY_DIR)), name="library")

# Serve FE static assets (HTML, CSS, JS dùng chung nếu cần)
app.mount("/static", StaticFiles(directory="../FE/static"), name="static")
