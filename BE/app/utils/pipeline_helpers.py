"""
Module: app.utils.pipeline_helpers
Description: Helper functions for the translation pipeline — VLM fallback call,
             font path discovery, text block pre-processing, key pool management,
             library path utilities, and per-chapter diagnostics.
"""

import asyncio
import re
import os
import logging
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image

from app.core.config import settings
from app.ai.base_model import BaseModel
from app.ai.fallback_engine import execute_fallback_chain
from app.ai.gemini import GEMINI_REQUESTS_PARALLEL_LIMITS
from app.utils.image import read_image_to_pil, compress_and_encode_image
from app.utils.grid_overlay import apply_grid_overlay
from app.utils.typesetting import find_optimal_typesetting
from app.services.inpainting import apply_solid_fill
from manga_translator.utils import TextBlock

logger = logging.getLogger(__name__)


# ===========================================================================
# VLM Fallback call
# ===========================================================================

async def call_vlm_with_fallback(
    primary_model_name: str,
    vlm_image_base64: str,
    genre: list[str],
    character_graph: str = "",
    previous_chapter_summary: str = "",
    preferred_key=None,
    single_model_only: bool = False,   # True = async mode: không fallback sang model khác
    manga_name: str = "",
    chapter_number: int = 0,
) -> tuple[dict[str, Any], BaseModel]:
    """
    Executes the VLM API call using the advanced FallbackEngine.

    single_model_only=True: chỉ thử primary_model_name — dùng cho translate_async.
    """
    return await execute_fallback_chain(
        primary_model_name=primary_model_name,
        vlm_image_base64=vlm_image_base64,
        genre=genre,
        character_graph=character_graph,
        previous_chapter_summary=previous_chapter_summary,
        preferred_key=preferred_key,
        single_model_only=single_model_only,
        manga_name=manga_name,
        chapter_number=chapter_number,
    )


# ===========================================================================
# Model rate-limit config
# ===========================================================================

_DEFAULT_LIMIT: dict = {"cooldown_seconds": 10}


def get_model_limit(model: str) -> dict:
    """Trả về cooldown config cho model. Prefix-match để xử lý alias."""
    for key, cfg in GEMINI_REQUESTS_PARALLEL_LIMITS.items():
        if model.startswith(key) or key.startswith(model):
            return cfg
    return _DEFAULT_LIMIT


# ===========================================================================
# Library path utilities
# CWD = BE/ sau khi run.py thực hiện chdir
# ===========================================================================

_LIBRARY_DIR = Path("../library")


def make_library_path(manga_name: str, manga_id: str, chapter_number: int) -> Path:
    """
    Compute đường dẫn output cho 1 chapter dịch trong library/.

    Cấu trúc: library/{manga_name} [{manga_id[:8]}]/{chapter_number}/
    Tên folder được sanitize để an toàn trên filesystem.
    """
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", manga_name).strip() or "Unknown"
    folder    = f"{safe_name} [{manga_id[:8]}]"
    return _LIBRARY_DIR / folder / str(chapter_number)


def make_library_url(manga_name: str, manga_id: str, chapter_number: int, filename: str) -> str:
    """Compute URL /library/... cho browser load ảnh từ server."""
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", manga_name).strip() or "Unknown"
    folder    = f"{safe_name} [{manga_id[:8]}]"
    return f"/library/{folder}/{chapter_number}/{filename}"


# ===========================================================================
# Key Pool — Token bucket rotation cho mọi tier
#
# Mỗi API key là 1 "token" trong asyncio.Queue.
# Chỉ 1 coroutine giữ 1 key tại 1 thời điểm → không bao giờ 429 burst.
#   tier="free" → GEMINI_API_KEYS     (Async + Sync mode)
#   tier="pro"  → GEMINI_PRO_API_KEYS (Rich Mode)
# ===========================================================================

_KEY_POOLS: dict[str, asyncio.Queue] = {}


def get_key_pool(tier: str = "free") -> asyncio.Queue:
    """
    Lazy-init Key Pool cho tier được chọn.

    tier="free" (mặc định) — GEMINI_API_KEYS, dùng cho Async + Sync.
    tier="pro"             — GEMINI_PRO_API_KEYS, dùng cho Rich Mode.
    """
    if tier in _KEY_POOLS:
        return _KEY_POOLS[tier]

    pool = asyncio.Queue()

    if tier == "free":
        from app.ai.key_manager import KeyManager
        keys = KeyManager().keys
        for key in keys:
            pool.put_nowait(key)
        logger.info(f"[KeyPool:free] ✨ {len(keys)} free key(s) — Async + Sync mode.")

    elif tier == "pro":
        from app.ai.key_manager import Key
        raw_keys = [k for k in settings.GEMINI_PRO_API_KEYS if k.strip()]
        for i, raw_key in enumerate(raw_keys):
            pool.put_nowait(Key(raw_key, key_id=i + 1))
        if raw_keys:
            logger.info(f"[KeyPool:pro] ✨ {len(raw_keys)} pro key(s) — Rich Mode.")
        else:
            logger.warning("[KeyPool:pro] ⚠️ GEMINI_PRO_API_KEYS trống — Rich Mode sẽ bị treo.")

    else:
        raise ValueError(f"[KeyPool] Unknown tier: {tier!r}")

    _KEY_POOLS[tier] = pool
    return pool


def schedule_key_return(pool: asyncio.Queue, key, delay: float) -> None:
    """
    Trả key về pool sau `delay` giây (non-blocking).

    Spawn background task thay vì await để caller không bị block.
    """
    async def _return():
        await asyncio.sleep(max(0.0, delay))
        pool.put_nowait(key)
    asyncio.create_task(_return())


def seconds_until_reset(key) -> float:
    """
    Số giây đến khi key được reset daily quota (VN UTC+7).
    Tối thiểu 3600s để tránh trường hợp reset_day NULL/bug.
    """
    vn_tz = timezone(timedelta(hours=7))
    reset  = key.reset_day if key.reset_day else key.calculate_next_reset()
    delta  = (reset - datetime.now(vn_tz)).total_seconds()
    return max(3600.0, delta)


# ===========================================================================
# Image preparation
# ===========================================================================

def prepare_image(file_bytes: bytes) -> tuple[Image.Image, str, str]:
    """
    Đọc ảnh từ bytes, nén và apply grid overlay cho VLM.

    Returns: (orig_image, image_base64, vlm_image_base64)
      - orig_image      — PIL Image gốc (full resolution)
      - image_base64    — ảnh nén base64 (dùng cho preview)
      - vlm_image_base64 — ảnh với grid overlay (gửi cho VLM)
    """
    orig_image = read_image_to_pil(file_bytes)
    try:
        image_base64 = compress_and_encode_image(
            orig_image,
            max_size=settings.MAX_IMAGE_SIZE,
            quality=80,
        )
    except Exception as img_err:
        logger.warning(f"Image compression failed, falling back to raw encode: {img_err}")
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    try:
        _, vlm_image_base64 = apply_grid_overlay(orig_image)
        logger.debug("Grid overlay applied.")
    except Exception as grid_err:
        logger.warning(f"Grid overlay failed (non-fatal): {grid_err}")
        vlm_image_base64 = image_base64

    return orig_image, image_base64, vlm_image_base64


# ===========================================================================
# Timing logger
# ===========================================================================

def log_timings(timings: dict, has_dialogue: bool, page_index: int) -> None:
    """Chi tiết thời gian từng bước — DEBUG only."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    parts = [
        f"img={timings['image_compression']:.2f}s",
        f"vlm={timings['vlm_api_call']:.2f}s",
    ]
    if has_dialogue:
        parts += [
            f"typeset={timings['bubble_analysis_typesetting']:.2f}s",
            f"inpaint={timings['inpainting']:.2f}s",
            f"render={timings['rendering']:.2f}s",
        ]
    parts.append(f"total={timings['total']:.2f}s")
    logger.debug(f"[Page {page_index}] timings: {' | '.join(parts)}")


# ===========================================================================
# DB helpers
# ===========================================================================

async def fetch_previous_chapter_summary(
    manga_id: str,
    chapter_number: int,
) -> str:
    """
    Truy xuất summary của chapter trước từ DB để inject vào prompt.

    Logic:
      - chapter <= 1 → đây là chapter đầu tiên → trả ""
      - chapter >= 2 → query summary của chapter (chapter_number - 1)
      - Không tìm thấy → trả "" (chưa dịch chapter trước)
    """
    if chapter_number <= 1:
        return ""

    from app.services.db_service import AsyncSessionLocal, ChapterSummaryCRUD
    prev_chapter = chapter_number - 1
    try:
        async with AsyncSessionLocal() as session:
            record = await ChapterSummaryCRUD.get(session, manga_id, prev_chapter)
            if record and record.summary:
                logger.debug(
                    f"Previous chapter summary loaded: ch={prev_chapter}, "
                    f"{len(record.summary)} chars."
                )
                return record.summary
        return ""
    except Exception as e:
        logger.warning(f"Failed to fetch previous chapter summary (non-fatal): {e}")
        return ""


# ===========================================================================
# VLM error constructor
# ===========================================================================

def make_vlm_error(page_index: int, reason: str = "unknown") -> dict:
    """Tạo VLM error result chuẩn cho trang không dịch được."""
    return {"page_index": page_index, "error": reason}


# ===========================================================================
# Key statistics monitoring (per-chapter)
# ===========================================================================

def reset_key_stats() -> None:
    """Reset request_count trên tất cả key trước mỗi chapter mới."""
    from app.ai.key_manager import KeyManager
    for k in KeyManager().keys:
        k.request_count = 0


def log_key_stats(chapter_label: str) -> None:
    """
    Log số request mỗi API key đã nhận — để monitor phân phối tải và debug RPM.
    Format: K#1[...XXXX]=3, K#2[...YYYY]=5, ...
    Chỉ log key có request_count > 0 để tránh noise.
    """
    from app.ai.key_manager import KeyManager
    km    = KeyManager()
    stats = {
        f"K#{i+1}[...{k.key[-4:]}]": k.request_count
        for i, k in enumerate(km.keys)
        if k.request_count > 0
    }
    total_reqs = sum(stats.values())
    if stats:
        logger.info(f"[{chapter_label}] 📊 Key stats ({total_reqs} API calls): {stats}")
    else:
        logger.info(f"[{chapter_label}] 📊 Key stats: no requests recorded.")


# ===========================================================================
# Font path discovery
# ===========================================================================

def get_font_path() -> str:
    """Selects the first valid font path from configuration settings."""
    font_path = settings.DEFAULT_FONT_PATH
    if not os.path.exists(font_path):
        for f in settings.FALLBACK_FONTS:
            if os.path.exists(f):
                font_path = f
                break
    return font_path


# ===========================================================================
# Text block pre-processing
# ===========================================================================

def prepare_text_blocks(
    translations: list[dict],
    orig_image: Image.Image,
    vlm_model_used: BaseModel,
) -> tuple[list[TextBlock], list[np.ndarray], np.ndarray]:
    """
    Processes translation boxes from VLM output into renderable TextBlocks.
    Uses exact schema keys: box_2d, text_vi, clr, direction.
    """
    text_regions = []
    full_regions_pts = []
    W_orig, H_orig = orig_image.size
    img_rgb = np.array(orig_image.convert("RGB"))

    for box_item in translations:
        if not isinstance(box_item, dict):
            continue

        # Extract coordinate — exact key from schema
        box_2d = box_item.get("box_2d")
        if not box_2d or not isinstance(box_2d, list) or len(box_2d) != 4:
            continue

        ymin, xmin, ymax, xmax = box_2d

        # Safely convert to normalized integers [0, 1000]
        try:
            ymin = max(0, min(1000, int(ymin)))
            xmin = max(0, min(1000, int(xmin)))
            ymax = max(0, min(1000, int(ymax)))
            xmax = max(0, min(1000, int(xmax)))
        except (ValueError, TypeError):
            continue

        # Prevent zero-size bounding boxes to prevent division by zero in renderer
        if ymax - ymin < 15:
            if ymax + 15 <= 1000:
                ymax = ymin + 15
            else:
                ymin = max(0, ymax - 15)

        if xmax - xmin < 15:
            if xmax + 15 <= 1000:
                xmax = xmin + 15
            else:
                xmin = max(0, xmax - 15)

        # Scale to original pixels
        x1 = int(xmin * W_orig / 1000)
        y1 = int(ymin * H_orig / 1000)
        x2 = int(xmax * W_orig / 1000)
        y2 = int(ymax * H_orig / 1000)

        # Solid fill region padding
        pad_sol_x = 2
        pad_sol_top = 2
        pad_sol_bottom = 5
        x1_sol = max(0, x1 - pad_sol_x)
        y1_sol = max(0, y1 - pad_sol_top)
        x2_sol = min(W_orig, x2 + pad_sol_x)
        y2_sol = min(H_orig, y2 + pad_sol_bottom)
        pts_solid = np.array(
            [[x1_sol, y1_sol], [x2_sol, y1_sol], [x2_sol, y2_sol], [x1_sol, y2_sol]],
            dtype=np.int32,
        )

        # LaMa inpainting region padding
        pad_x = 5
        pad_top = 5
        pad_bottom = 10
        x1_lam = max(0, x1 - pad_x)
        y1_lam = max(0, y1 - pad_top)
        x2_lam = min(W_orig, x2 + pad_x)
        y2_lam = min(H_orig, y2 + pad_bottom)
        pts_lama = np.array(
            [[x1_lam, y1_lam], [x2_lam, y1_lam], [x2_lam, y2_lam], [x1_lam, y2_lam]],
            dtype=np.int32,
        )

        # Shrink drawing coordinates to 82% to prevent text border clipping
        centroid  = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        pts_tight = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
        )
        pts_shrunk = (centroid + (pts_tight - centroid) * 0.82).astype(np.int32)

        # Retrieve Vietnamese translated text — exact key from schema
        vietnamese_text = box_item.get("text_vi", "")
        if not isinstance(vietnamese_text, str) or not vietnamese_text.strip():
            continue

        box_width  = x2 - x1
        box_height = y2 - y1

        # Background color classification — exact key from schema
        vlm_clr = box_item.get("clr")
        if vlm_clr is not None:
            try:
                vlm_clr = int(vlm_clr)
            except (ValueError, TypeError):
                vlm_clr = None

        median_color = np.array([255, 255, 255], dtype=np.uint8)
        is_uniform   = False

        if vlm_clr == 1:
            is_uniform   = True
            median_color = np.array([255, 255, 255], dtype=np.uint8)
        elif vlm_clr == 3:
            is_uniform   = True
            median_color = np.array([0, 0, 0], dtype=np.uint8)
        elif vlm_clr == 2:
            is_uniform = False
        else:
            # Mathematical boundary fallback analysis
            if box_height > 4 and box_width > 4:
                crop          = img_rgb[y1:y2, x1:x2]
                border_pixels = []
                border_pixels.extend(crop[0, :])
                border_pixels.extend(crop[-1, :])
                border_pixels.extend(crop[1:-1, 0])
                border_pixels.extend(crop[1:-1, -1])
                border_pixels = np.array(border_pixels)

                if len(border_pixels) > 0:
                    median_color = np.median(border_pixels, axis=0)
                    std_color    = np.std(border_pixels, axis=0)
                    max_std      = np.max(std_color)

                    if max_std < 22.0:
                        is_uniform = True
                    elif np.all(median_color > 235) and max_std < 30.0:
                        is_uniform = True
                    elif np.all(median_color < 30) and max_std < 30.0:
                        is_uniform = True

        if is_uniform:
            img_rgb = apply_solid_fill(img_rgb, pts_solid, median_color)
        else:
            full_regions_pts.append(pts_lama)

        # Decide text colors based on background brightness
        bg_brightness = np.mean(median_color)
        if bg_brightness < 127:
            fg_color = (255, 255, 255)
            bg_color = (0, 0, 0)
        else:
            fg_color = (0, 0, 0)
            bg_color = (255, 255, 255)

        # Dynamic typesetting layout optimization
        shrunk_width  = int(box_width  * 0.82)
        shrunk_height = int(box_height * 0.82)

        optimal_font_size, needed_rows, balanced_lines = find_optimal_typesetting(
            vietnamese_text, shrunk_width, shrunk_height
        )

        vietnamese_text_with_newlines = "\n".join(balanced_lines)
        texts_placeholder             = [""] * needed_rows

        region = TextBlock(
            lines=[pts_shrunk],
            texts=texts_placeholder,
            font_size=optimal_font_size,
            angle=0,
            translation=vietnamese_text_with_newlines,
            fg_color=fg_color,
            bg_color=bg_color,
            language='ja',
            target_lang='vi',
            direction='h',
        )
        region.text_raw = ""
        text_regions.append(region)

    logger.info(f"Successfully processed {len(text_regions)} text regions.")
    return text_regions, full_regions_pts, img_rgb
