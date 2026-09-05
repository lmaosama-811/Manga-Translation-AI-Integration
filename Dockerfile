# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Manga VLM Translator
#
# Image này dùng chung cho 3 services: api, worker, beat.
# Lý do: cùng codebase Python, chỉ khác lệnh khởi động (CMD trong compose).
#
# WORKDIR = /app/BE vì toàn bộ app code dùng path tương đối:
#   "../FE/static" , "../library" , "fonts/" — đều được resolve từ BE/
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Manga VLM Translator"
LABEL org.opencontainers.image.description="FastAPI + Celery + LaMa inpainting + Gemini VLM"

# ── System dependencies ────────────────────────────────────────────────────────
# git         : cần để pip cài pydensecrf từ GitHub URL
# libgl1      : OpenCV (cv2) cần thư viện OpenGL
# libglib2.0  : OpenCV cần glib
# libsm6 / libxext6 / libxrender1 : OpenCV headless deps
# libgomp1    : PyTorch parallel ops (OpenMP)
# curl        : dùng cho healthcheck của api container
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements.txt TRƯỚC khi copy source code.
# Docker layer cache: nếu requirements.txt không đổi → layer này được cache lại
# → không cần cài lại toàn bộ packages khi chỉ sửa code.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Playwright Chromium ────────────────────────────────────────────────────────
# Cài Chromium + toàn bộ system deps của nó (--with-deps).
# Dùng cho tính năng scrape URL (browser_scraper.py).
RUN playwright install --with-deps chromium

# ── Source code ────────────────────────────────────────────────────────────────
# CHỈ copy BE/ và FE/ — KHÔNG copy:
#   - BE/models/  → mount volume từ máy host (217MB model weights)
#   - BE/fonts/   → mount volume từ máy host
#   - library/    → mount volume từ máy host (ảnh đã dịch)
#   - .env        → KHÔNG bao giờ bake secrets vào image
COPY BE/ ./BE/
COPY FE/ ./FE/

# Tạo các mount point (thư mục rỗng để volume attach vào đúng chỗ)
RUN mkdir -p library BE/models BE/fonts

# ── Working directory ──────────────────────────────────────────────────────────
# PHẢI là /app/BE vì:
#   config.py  : env_file="../.env"      → /app/.env        ✓
#   main.py    : Path("../FE/static")    → /app/FE/static   ✓
#   main.py    : Path("../library")      → /app/library      ✓
WORKDIR /app/BE

# ── Default command (override trong docker-compose per service) ────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
