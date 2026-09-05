"""
Module: app.utils.pipeline_helpers
Description: Helper functions for the translation pipeline — VLM fallback call,
             font path discovery, text block pre-processing, key pool management,
             library path utilities, and per-chapter diagnostics.

Preprocessing (prepare_image):
  Trước: apply Grid Overlay 10×10 → Gemini đọc tick marks → trả về box_2d (0-1000)
  Sau:  RT-DETR-v2 phát hiện bubble pixel-perfect → vẽ bubble_id số đỏ → Gemini đọc số
"""

import asyncio
import re
import os
import logging
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image

from app.core.config import settings
from app.ai.base_model import BaseModel
from app.ai.fallback_engine import execute_fallback_chain
from app.ai.gemini import GEMINI_REQUESTS_PARALLEL_LIMITS
from app.utils.image import read_image_to_pil, compress_and_encode_image, encode_pil_to_base64
from manga_translator.rendering.ballon_extractor import extract_ballon_region
# from app.services.inpainting import apply_solid_fill
from manga_translator.utils import TextBlock

logger = logging.getLogger(__name__)


# ===========================================================================
# VLM Fallback call
# ===========================================================================

async def call_vlm_with_fallback(
    primary_model_name: str,
    vlm_image_base64: str,
    genre: list[str],
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
# Bubble annotation helper (Compact labels on tight text bbox xyxy)
# ===========================================================================

def annotate_bubbles_with_numbers(
    img_rgb: np.ndarray,
    text_blocks: list,
) -> np.ndarray:
    """
    Vẽ nhãn số màu đỏ đậm (1, 2, 3...) tại góc trên-trái của từng TEXT BBOX (xyxy).
    Nền hình chữ nhật trắng nhỏ giúp số nổi bật trên mọi loại nền.

    Args:
        img_rgb:     Ảnh RGB (H, W, 3) numpy array.
        text_blocks: Danh sách các đối tượng TextBlock đã có bubble_id (1..N).
    Returns:
        Ảnh RGB đã vẽ nhãn số.
    """
    annotated = img_rgb.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 2

    for blk in text_blocks:
        if getattr(blk, "xyxy", None) is None:
            continue
        x1, y1, _, _ = blk.xyxy
        bubble_id = getattr(blk, "bubble_id", None)
        if bubble_id is None:
            continue

        label = str(bubble_id)
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Vị trí PHÍA TRÊN góc trên-trái của ruột chữ (xyxy) để không đè lên chữ
        badge_w = text_w + 6
        badge_h = text_h + 6

        bg_x1 = max(0, int(x1))
        bg_x2 = min(annotated.shape[1], bg_x1 + badge_w)

        # Đặt phía trên y1. Nếu y1 nằm sát mép trên ảnh thì đặt ngay y1
        if int(y1) - badge_h >= 0:
            bg_y2 = int(y1) - 2
            bg_y1 = bg_y2 - badge_h
        else:
            bg_y1 = max(0, int(y1))
            bg_y2 = bg_y1 + badge_h

        # Nền trắng nhỏ gọn + viền đỏ nhẹ
        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (210, 0, 0), 1)

        # Số đỏ đậm
        cv2.putText(
            annotated, label,
            (bg_x1 + 3, bg_y2 - 3),
            font, font_scale, (210, 0, 0), thickness, cv2.LINE_AA,
        )

    return annotated


# ===========================================================================
# Image preparation
# ===========================================================================

def prepare_image(
    file_bytes: bytes,
) -> tuple[Image.Image, str, str, list]:
    """
    Đọc ảnh, chạy RT-DETR-v2 PyTorch FP32 detector, vẽ bubble_id trên vùng chữ (xyxy) và trả về.

    Returns:
        (orig_image, image_base64, vlm_image_base64, detected_text_blocks)
          - orig_image           — PIL Image gốc (full resolution).
          - image_base64         — ảnh nén base64 (dùng cho preview).
          - vlm_image_base64      — ảnh với bubble_id annotation thắt chặt trên xyxy.
          - detected_text_blocks — danh sách đối tượng TextBlock đầy đủ thông tin.
    """
    orig_image = read_image_to_pil(file_bytes)
    img_rgb    = np.array(orig_image.convert("RGB"))

    # ── Bước 1: Chạy RT-DETR-v2 PyTorch FP32 detector ───────────────────────
    detected_text_blocks: list = []
    try:
        from app.detection.rt_detr import get_detector
        detector = get_detector()
        detected_text_blocks = detector.detect(img_rgb)
        logger.info(f"[Detector FP32] Phát hiện {len(detected_text_blocks)} khối chữ (TextBlock).")
    except Exception as det_err:
        logger.error(f"[Detector FP32] Detection thất bại: {det_err}", exc_info=True)

    # ── Bước 2: Nén ảnh gốc → preview base64 ─────────────────────────────
    try:
        image_base64 = compress_and_encode_image(
            orig_image,
            max_size=settings.MAX_IMAGE_SIZE,
            quality=80,
        )
    except Exception as img_err:
        logger.warning(f"Image compression failed, falling back to raw encode: {img_err}")
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    # ── Bước 3: Vẽ bubble_id trên vùng chữ (xyxy) → vlm base64 ─────────────
    try:
        if detected_text_blocks:
            annotated_rgb = annotate_bubbles_with_numbers(img_rgb, detected_text_blocks)
        else:
            annotated_rgb = img_rgb

        annotated_pil   = Image.fromarray(annotated_rgb)
        vlm_image_base64 = encode_pil_to_base64(annotated_pil, quality=85)
        logger.debug("Bubble ID annotation applied to VLM image.")
    except Exception as ann_err:
        logger.warning(f"Bubble annotation failed (non-fatal): {ann_err}")
        vlm_image_base64 = image_base64

    return orig_image, image_base64, vlm_image_base64, detected_text_blocks



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
    detected_blocks: list | None = None,
) -> tuple[list, list[np.ndarray], np.ndarray]:
    """
    Xử lý kết quả dịch từ Gemini VLM thành danh sách TextBlock để render.

    Sử dụng Dual-Box từ RT-DETR-v2 FP32 Detector:
      - `matched_block.xyxy`: Tọa độ vùng chữ thắt chặt (dùng cho Typesetting & Render).
      - `matched_block.bubble_xyxy`: Tọa độ bong bóng thoại bao ngoài (dùng cho LaMa Inpainting).
      - `matched_block.direction`: Hướng chữ ('horizontal' | 'vertical' từ heuristic_lines).
      - `matched_block.font_color`: Màu chữ gốc (từ Border-Otsu).
    """
    text_regions:     list = []
    full_regions_pts: list[np.ndarray] = []
    W_orig, H_orig = orig_image.size
    img_rgb = np.array(orig_image.convert("RGB"))

    blocks_map = {}
    if detected_blocks:
        for idx, blk in enumerate(detected_blocks, start=1):
            bid = getattr(blk, "bubble_id", idx)
            blocks_map[bid] = blk

    for box_item in translations:
        if not isinstance(box_item, dict):
            continue

        vietnamese_text = box_item.get("text_vi", "")
        if not isinstance(vietnamese_text, str) or not vietnamese_text.strip():
            continue
        vietnamese_text = vietnamese_text.strip()

        matched_blk = None
        bubble_id = box_item.get("bubble_id")
        if bubble_id is not None:
            try:
                matched_blk = blocks_map.get(int(bubble_id))
            except (ValueError, TypeError):
                pass

        if matched_blk is not None and getattr(matched_blk, "xyxy", None) is not None:
            # ── Lấy tọa độ từ PyTorch FP32 Detector ─────────────────────────
            # Render box: xyxy thắt chặt ruột chữ
            x1, y1, x2, y2 = [int(v) for v in matched_blk.xyxy]

            # Inpainting box: bubble_xyxy (bong bóng bao ngoài) nếu có, else xyxy
            if getattr(matched_blk, "bubble_xyxy", None) is not None:
                ix1, iy1, ix2, iy2 = [int(v) for v in matched_blk.bubble_xyxy]
            else:
                ix1, iy1, ix2, iy2 = x1, y1, x2, y2

            det_direction = getattr(matched_blk, "direction", "horizontal")
            direction_val = "v" if det_direction == "vertical" else "h"
        else:
            # Fallback: box_2d 0-1000 scale
            box_2d = box_item.get("box_2d")
            if not box_2d or not isinstance(box_2d, list) or len(box_2d) != 4:
                continue

            try:
                ymin = max(0, min(1000, int(box_2d[0])))
                xmin = max(0, min(1000, int(box_2d[1])))
                ymax = max(0, min(1000, int(box_2d[2])))
                xmax = max(0, min(1000, int(box_2d[3])))
            except (ValueError, TypeError):
                continue

            x1 = int(xmin * W_orig / 1000)
            y1 = int(ymin * H_orig / 1000)
            x2 = int(xmax * W_orig / 1000)
            y2 = int(ymax * H_orig / 1000)
            direction_val = "h"

        # Clamp tọa độ trong khổ ảnh gốc
        x1, x2 = max(0, min(W_orig, x1)), max(0, min(W_orig, x2))
        y1, y2 = max(0, min(H_orig, y1)), max(0, min(H_orig, y2))
        ix1, ix2 = max(0, min(W_orig, ix1)), max(0, min(W_orig, ix2))
        iy1, iy2 = max(0, min(H_orig, iy1)), max(0, min(H_orig, iy2))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        box_width  = x2 - x1
        box_height = y2 - y1

        # ── Bubble mask (comic-translate's extract_ballon_region — 100% identical) ────────
        bubble_xywh = [ix1, iy1, ix2 - ix1, iy2 - iy1]
        ballon_mask, ballon_xyxy = extract_ballon_region(img_rgb, bubble_xywh, enlarge_ratio=1)

        # LaMa region pts vẫn giữ nguyên cho non-uniform background
        pts_lama = np.array(
            [[max(0, ix1 - 5), max(0, iy1 - 5)], [min(W_orig, ix2 + 5), max(0, iy1 - 5)],
             [min(W_orig, ix2 + 5), min(H_orig, iy2 + 10)], [max(0, ix1 - 5), min(H_orig, iy2 + 10)]],
            dtype=np.int32,
        )

        # ── Vùng vẽ chữ (xyxy thắt chặt ruột chữ) ─────────────────────────
        centroid   = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        pts_tight  = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        pts_shrunk = (centroid + (pts_tight - centroid) * 0.88).astype(np.int32)

        # ── Màu nền (clr) ─────────────────────────────────────────────────
        vlm_clr = box_item.get("clr")
        try:
            vlm_clr = int(vlm_clr) if vlm_clr is not None else None
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
            if iy2 - iy1 > 4 and ix2 - ix1 > 4:
                crop = img_rgb[iy1:iy2, ix1:ix2]
                border_pixels = []
                border_pixels.extend(crop[0, :])
                border_pixels.extend(crop[-1, :])
                border_pixels.extend(crop[1:-1, 0])
                border_pixels.extend(crop[1:-1, -1])
                border_pixels = np.array(border_pixels)
                if len(border_pixels) > 0:
                    median_color = np.median(border_pixels, axis=0)
                    max_std      = np.max(np.std(border_pixels, axis=0))
                    if max_std < 25.0:
                        is_uniform = True

        # ── Màu chữ: Border-Otsu từ rt_detr (100% giống comic-translate) ────
        # Phải extract TRƯỚC apply_solid_fill để thấy pixel chữ thực tế
        from app.detection.rt_detr import extract_foreground_color as rt_extract_fg
        crop_rgb_for_color = img_rgb[y1:y2, x1:x2]
        fg_extracted = rt_extract_fg(crop_rgb_for_color)
        if fg_extracted is not None:
            fg_color = tuple(fg_extracted)
        else:
            bg_brightness = float(np.mean(median_color))
            fg_color = (0, 0, 0) if bg_brightness > 127 else (255, 255, 255)
        fg_brightness = 0.299 * fg_color[0] + 0.587 * fg_color[1] + 0.114 * fg_color[2]
        bg_color = (255, 255, 255) if fg_brightness < 127 else (0, 0, 0)

        # ── text_class ────────────────────────────────────────────────────────
        text_cls    = getattr(matched_blk, "text_class", "text_bubble") if matched_blk else "text_bubble"
        alignment_val = "center" if text_cls == "text_bubble" else "left"

        # ── Build render region (comic-translate: blk with .xyxy, .bubble_xyxy, .translation) ──
        # comic-translate pipeline (batch_processor.py):
        #   1. blk.xyxy        = tight text bbox from detector
        #   2. blk.bubble_xyxy = outer bubble bbox
        #   3. get_best_render_area() → blk.xyxy = shrink_bbox(bubble_xyxy, 0.05)  ← for text_bubble
        #   4. pil_word_wrap(width=blk.xywh[2], height=blk.xywh[3], ...)           ← binary search
        #
        # We store bubble_xyxy and text_class so get_best_render_area() can apply the shrink.
        # xyxy must be a mutable numpy array (blk.xyxy[:] = [...] in get_best_render_area)
        region_xyxy        = np.array([x1, y1, x2, y2], dtype=np.int32)
        region_bubble_xyxy = np.array([ix1, iy1, ix2, iy2], dtype=np.int32) if matched_blk is not None and getattr(matched_blk, "bubble_xyxy", None) is not None else None

        # Max font = bubble height / 3 capped [14, 40] — generous initial for pil_word_wrap
        if region_bubble_xyxy is not None:
            bub_h = int(region_bubble_xyxy[3] - region_bubble_xyxy[1])
        else:
            bub_h = box_height
        max_font = max(18, min(int(bub_h / 2.5), 52))
        min_font = 10   # comic-translate default

        class _RenderRegion:
            """Lightweight object matching the interface expected by draw_text() and get_best_render_area()."""
            __slots__ = (
                "xyxy", "bubble_xyxy", "text_class", "translation",
                "alignment", "font_color", "min_font_size", "max_font_size",
                "source_lang_direction",
            )

        rr = _RenderRegion()
        rr.xyxy               = region_xyxy            # mutable — get_best_render_area will overwrite
        rr.bubble_xyxy        = region_bubble_xyxy
        rr.text_class         = text_cls
        rr.translation        = vietnamese_text
        rr.alignment          = alignment_val
        rr.font_color         = fg_color               # tuple (R,G,B)
        rr.min_font_size      = min_font
        rr.max_font_size      = max_font
        rr.source_lang_direction = getattr(matched_blk, "direction", "horizontal") if matched_blk else "horizontal"

        text_regions.append(rr)

    logger.info(
        f"prepare_text_blocks: {len(text_regions)} text region(s) — "
        f"{len(full_regions_pts)} LaMa region(s)."
    )
    return text_regions, full_regions_pts, img_rgb

