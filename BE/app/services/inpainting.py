"""
Module: app.services.inpainting
Description: Provides high-efficiency bubble clearing workflows (instant solid fills + crop-region AI LaMa inpainting).

Concurrency model:
  - Global Semaphore(1) ensures at most 1 LaMa inference runs at any time across all users.
  - asyncio.to_thread wraps the blocking PyTorch forward pass, keeping the event loop
    responsive while User B is waiting for the semaphore.
"""

# pyrefly: ignore [missing-import]
import cv2
import asyncio
import numpy as np
import torch
import logging

from manga_translator.config import Config
from manga_translator.inpainting import dispatch as dispatch_inpainting, get_inpainter
from manga_translator.inpainting.common import OfflineInpainter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Global LaMa Semaphore
# Serialize ALL inpainting across users — only 1 LaMa forward pass at a time.
#
# Why Semaphore(1) and not just sequential code?
#   translate_sync Phase 2 is sequential *within* a single request, but
#   two concurrent translate_sync calls (2 users) would both reach Phase 2.
#   Without a semaphore they'd compete for CPU/GPU causing slowdown or OOM.
#
# Why asyncio.to_thread?
#   LaMa's _infer() is async def but has NO internal await → it blocks the
#   event loop. to_thread runs it in a worker thread so User B can properly
#   await the semaphore instead of being frozen by User A's torch computation.
# ------------------------------------------------------------------

_LAMA_GLOBAL_SEM: asyncio.Semaphore | None = None


def _get_lama_semaphore() -> asyncio.Semaphore:
    """
    Lazy-init global LaMa semaphore — Semaphore(1).
    Thread-safe within asyncio single-threaded event loop.
    """
    global _LAMA_GLOBAL_SEM
    if _LAMA_GLOBAL_SEM is None:
        _LAMA_GLOBAL_SEM = asyncio.Semaphore(1)
        logger.info("[LaMa] Global semaphore(1) initialized — inpainting serialized across all requests.")
    return _LAMA_GLOBAL_SEM


def apply_solid_fill(img_rgb: np.ndarray, pts_solid: np.ndarray, color: np.ndarray) -> np.ndarray:
    """
    Erases speech text by mathematically filling the bubble region with its median solid color.
    """
    fill_color = np.round(color).astype(np.uint8).tolist()
    cv2.fillPoly(img_rgb, [pts_solid], fill_color)
    return img_rgb


# ------------------------------------------------------------------
# Sync helper — called via asyncio.to_thread
# ------------------------------------------------------------------

def _run_region_sync(
    inpainter_key,
    crop_img: np.ndarray,
    crop_mask: np.ndarray,
    config,
    crop_inpainting_size: int,
    device: str,
) -> np.ndarray:
    """
    Sync wrapper for per-region LaMa — meant to be called via asyncio.to_thread.

    Creates a fresh event loop in the worker thread to drive the async dispatch chain.
    Safe because:
      - Only 1 call at a time (Semaphore(1) upstream).
      - PyTorch inference is thread-safe with torch.no_grad() (read-only forward).
      - Model is already loaded before entering this function.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            dispatch_inpainting(
                inpainter_key,
                crop_img,
                crop_mask,
                config,
                crop_inpainting_size,
                device,
                verbose=False,
            )
        )
    finally:
        loop.close()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def run_local_lama_inpainting(img_rgb: np.ndarray, full_regions_pts: list[np.ndarray]) -> np.ndarray:
    """
    Cleans up complex background/screentone regions by running local cropped-region LaMa inpainting.

    Flow:
      1. Pre-load model (idempotent — only loads weights on first call).
      2. Acquire global Semaphore(1) — serialize across concurrent requests.
      3. Per-region: crop → build mask → run LaMa in thread → stitch back.

    Avoids downscaling, preserves original resolution, speeds up to <0.2s per bubble.
    """
    if not full_regions_pts:
        return img_rgb

    img_inpainted = img_rgb.copy()
    W_orig, H_orig = img_rgb.shape[1], img_rgb.shape[0]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    config  = Config()
    inpainter_key = config.inpainter.inpainter

    # ── Pre-load model before acquiring the semaphore ──────────────────
    # load() is idempotent; only runs once then caches the model in memory.
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.load(device)

    # ── Acquire global semaphore ────────────────────────────────────────
    sem = _get_lama_semaphore()
    if sem.locked():
        logger.debug("[LaMa] Semaphore busy — waiting for previous inpainting to finish.")

    n = len(full_regions_pts)
    logger.info(f"[LaMa] Inpainting {n} region(s) on {device}.")

    async with sem:
        for i_reg, full_pts in enumerate(full_regions_pts):
            # ── Crop minimum bounding rect + padding ───────────────────
            x_box, y_box, w_box, h_box = cv2.boundingRect(full_pts)
            pad   = 32
            x1    = max(0, x_box - pad)
            y1    = max(0, y_box - pad)
            x2    = min(W_orig, x_box + w_box + pad)
            y2    = min(H_orig, y_box + h_box + pad)

            crop_img  = img_rgb[y1:y2, x1:x2]
            crop_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

            local_pts = full_pts - np.array([x1, y1])
            cv2.fillConvexPoly(crop_mask, local_pts, 255)

            crop_size = max(crop_img.shape[0], crop_img.shape[1])

            logger.debug(
                f"[LaMa] Region {i_reg + 1}/{n}: "
                f"crop {crop_img.shape[1]}×{crop_img.shape[0]} on {device}"
            )

            # ── Run LaMa in worker thread (non-blocking) ───────────────
            crop_inpainted = await asyncio.to_thread(
                _run_region_sync,
                inpainter_key,
                crop_img,
                crop_mask,
                config,
                crop_size,
                device,
            )

            # ── Stitch result back to canvas ───────────────────────────
            img_inpainted[y1:y2, x1:x2] = crop_inpainted

    return img_inpainted
