"""
Module: app.api.routes.translate
Description: Main orchestration engine for the FastAPI translation endpoints.
             Coordinates image prep, VLM inference, inpainting, and rendering.

Architecture — Hệ thống fallback 2 tầng:

  Thiết kế Queue:
    Mỗi API key là 1 token trong asyncio.Queue.
    Chỉ 1 request được giữ 1 key tại 1 thời điểm → không bao giờ collision.

  Phase 1 — VLM (parallel):
    Mỗi task được kiểm soát bởi worker coroutine:
      - await pool.get() ← chờ key rảnh
      - Gọi execute_fallback_chain với key đó
      - Retry với key mới nếu bị AllModelsRPMError / KeyFullyExhaustedError
      - Trả key về pool sau cooldown phù hợp

  Phase 2 — Render (sequential):
    LaMa inpainting + text rendering — blocking, không parallel hóa được.
"""

import asyncio
import time
import logging
import uuid
from pathlib import Path
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

from app.utils.image import encode_pil_to_base64
from app.utils.pipeline_helpers import (
    call_vlm_with_fallback,
    get_font_path,
    prepare_text_blocks,
    get_model_limit,
    make_library_path,
    make_library_url,
    get_key_pool,
    schedule_key_return,
    seconds_until_reset,
    prepare_image,
    log_timings,
    fetch_previous_chapter_summary,
    make_vlm_error,
    reset_key_stats,
    log_key_stats,
)
from app.services.inpainting import run_local_lama_inpainting
from app.services.rendering import init_rendering_engine, render_text_bubbles
from app.services.graph_service import GraphService
from app.ai.fallback_engine import AllModelsRPMError, KeyFullyExhaustedError

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------
# Phase 1: VLM — Image prep + VLM call only (no render, no side-effect)
# ------------------------------------------------------------------

async def _vlm_single_page(
    image_bytes: bytes,
    *,
    manga_id: str,
    page_index: int,
    model: str,
    genre_list: list[str],
    previous_chapter_summary: str,
    preferred_key=None,
    single_model_only: bool = False,
    manga_name: str = "",
    chapter_number: int = 0,
) -> dict:
    """
    Phase 1: Chạy image prep và VLM call cho 1 trang.
    Không render, không Celery — chỉ trả về raw VLM output + ảnh gốc.

    single_model_only=True (async mode):
      Chỉ thử model chỉ định, không fallback sang flash/2.5-flash.
      Khi bị RPM → raise AllModelsRPMError ngay → worker re-queue trang với key mới.
      Mục đích: không waste thêm 2 API calls (flash/2.5-flash) khi flash-lite đã bị RPM.
    """
    try:
        t0 = time.perf_counter()
        orig_image, _, vlm_image_base64 = prepare_image(image_bytes)
        t_compression = time.perf_counter() - t0

        G = await GraphService.load(manga_id, use_cache=True)
        character_graph_prompt = GraphService.to_prompt(G)

        t1 = time.perf_counter()
        vlm_response, vlm_model_used = await call_vlm_with_fallback(
            primary_model_name=model,
            vlm_image_base64=vlm_image_base64,
            genre=genre_list,
            character_graph=character_graph_prompt,
            previous_chapter_summary=previous_chapter_summary,
            preferred_key=preferred_key,
            single_model_only=single_model_only,
            manga_name=manga_name,
            chapter_number=chapter_number,
        )
        t_vlm = time.perf_counter() - t1

        page_summary      = vlm_response.get("page_summary", "")
        character_updates = vlm_response.get("character_updates", [])

        # Per-page detail — DEBUG only
        logger.debug(
            f"[p{page_index}] VLM done in {t_vlm:.1f}s — "
            f"has_dialogue={vlm_response.get('has_dialogue')} "
            f"bubbles={len(vlm_response.get('translations', []))} "
            f"char_updates={len(character_updates)}"
        )

        return {
            "page_index":          page_index,
            "orig_image":          orig_image,
            "vlm_response":        vlm_response,
            "vlm_model_used":      vlm_model_used,
            "t_image_compression": t_compression,
            "t_vlm_api_call":      t_vlm,
            "error":               None,
        }

    except (AllModelsRPMError, KeyFullyExhaustedError):
        # Re-raise để Tầng 2 worker trong translate_async bắt và xử lý:
        # - AllModelsRPMError  → re-queue trang, key cooldown dài hơn
        # - KeyFullyExhaustedError → key bị loại khỏi pool đến daily reset
        raise
    except Exception as err:
        logger.error(
            f"VLM failed on page {page_index} (manga={manga_id!r}, model={model}): {err}",
            exc_info=True,
        )
        return {"page_index": page_index, "error": str(err)}


# ------------------------------------------------------------------
# Phase 2: Render — LaMa inpainting + text rendering (sequential)
# ------------------------------------------------------------------

async def _render_single_page(vlm_result: dict) -> dict:
    """
    Phase 2: Nhận kết quả VLM đã có sẵn, chạy LaMa + render text.
    Blocking (PyTorch), chạy sequential — không gather.

    Returns:
        {
            "rendered_image": PIL.Image,
            "rendered_base64": str,
            "has_dialogue": bool,
            "translations": list,
            "page_summary": str,
            "character_updates": list,
            "pronoun_shift": list,
            "timings": dict,
            "model_used": str,
        }
    """
    if vlm_result.get("error"):
        raise RuntimeError(vlm_result["error"])

    orig_image    = vlm_result["orig_image"]
    vlm_response  = vlm_result["vlm_response"]
    vlm_model_used = vlm_result["vlm_model_used"]
    page_index    = vlm_result["page_index"]

    page_summary      = vlm_response.get("page_summary", "")
    character_updates = vlm_response.get("character_updates", [])
    pronoun_shift     = vlm_response.get("pronoun_shift", [])
    has_dialogue      = vlm_response.get("has_dialogue", False)
    translations      = vlm_response.get("translations", [])

    timings = {
        "image_compression":        vlm_result["t_image_compression"],
        "vlm_api_call":             vlm_result["t_vlm_api_call"],
        "bubble_analysis_typesetting": 0.0,
        "inpainting":               0.0,
        "rendering":                0.0,
        "base64_encoding":          0.0,
        "total":                    0.0,
    }

    rendered_image = orig_image

    if has_dialogue and translations:
        t_step = time.perf_counter()
        font_path = get_font_path()
        init_rendering_engine(font_path)
        text_regions, full_regions_pts, img_rgb = prepare_text_blocks(
            translations, orig_image, vlm_model_used
        )
        timings["bubble_analysis_typesetting"] = time.perf_counter() - t_step

        if text_regions:
            t_inp = time.perf_counter()
            img_inpainted = await run_local_lama_inpainting(img_rgb, full_regions_pts)
            timings["inpainting"] = time.perf_counter() - t_inp

            t_rend = time.perf_counter()
            img_rendered_arr = await render_text_bubbles(
                img_inpainted, text_regions, font_path, line_spacing=0.20
            )
            timings["rendering"] = time.perf_counter() - t_rend

            rendered_image = Image.fromarray(img_rendered_arr)
    else:
        logger.debug(f"[p{page_index}] has_dialogue=false — skip render.")

    t_step = time.perf_counter()
    rendered_base64 = encode_pil_to_base64(rendered_image, quality=90)
    timings["base64_encoding"] = time.perf_counter() - t_step

    timings["total"] = (
        timings["image_compression"] + timings["vlm_api_call"]
        + timings["bubble_analysis_typesetting"] + timings["inpainting"]
        + timings["rendering"] + timings["base64_encoding"]
    )
    log_timings(timings, has_dialogue, page_index)

    return {
        "rendered_image":    rendered_image,
        "rendered_base64":   rendered_base64,
        "has_dialogue":      has_dialogue,
        "translations":      translations,
        "page_summary":      page_summary,
        "character_updates": character_updates,
        "pronoun_shift":     pronoun_shift,
        "timings":           timings,
        "model_used":        vlm_model_used.model_name,
    }


# ------------------------------------------------------------------
# Backward-compat: _translate_single_page = VLM + Render sequentially
# Dùng cho Playground endpoint (không cần parallel)
# ------------------------------------------------------------------

async def _translate_single_page(
    image_bytes: bytes,
    *,
    manga_id: str,
    page_index: int = 0,
    chapter_number: int = 0,
    model: str = "gemini-3.5-flash",
    genre_list: list[str] | None = None,
    previous_chapter_summary: str = "",
    manga_name: str = "",
) -> dict:
    """Convenience wrapper: VLM → Render trong 1 call (dùng cho Playground)."""
    vlm_result = await _vlm_single_page(
        image_bytes,
        manga_id=manga_id,
        page_index=page_index,
        model=model,
        genre_list=genre_list or [],
        previous_chapter_summary=previous_chapter_summary,
        manga_name=manga_name,
        chapter_number=chapter_number,
    )
    return await _render_single_page(vlm_result)


# ------------------------------------------------------------------
# FastAPI Route — Playground endpoint (UploadFile)
# ------------------------------------------------------------------

@router.post("/translate")
async def translate_manga_page(
    files: list[UploadFile] = File(..., description="File(s) ảnh trang truyện (PNG, JPG, JPEG, WEBP)"),
    model: str = Query("gemini-3.5-flash", description="Tên model VLM sử dụng"),
    genre: str = Query("", description="Danh sách skill thể loại, phân tách bằng dấu phẩy"),
    chapter_summary: str = Query("", description="Summary của chapter TRƯỚC"),
    manga_id: str = Query("", description="Slug ID bộ truyện - tự tạo nếu bỏ trống"),
    chapter_number: int = Query(0, description="Số thứ tự chapter"),
    page_index: int = Query(0, description="Số thứ tự trang đầu tiên (tăng dần nếu nhiều trang)"),
):
    manga_id   = manga_id.strip() or str(uuid.uuid4().hex)
    genre_list = [g.strip() for g in genre.split(",") if g.strip()]

    VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp")
    for f in files:
        if not (f.filename or "").lower().endswith(VALID_EXTS):
            raise HTTPException(
                status_code=400,
                detail=f"Chỉ hỗ trợ PNG, JPG, JPEG, WEBP. File {f.filename!r} không hợp lệ.",
            )

    try:
        if len(files) == 1:
            file_bytes = await files[0].read()
            result = await _translate_single_page(
                file_bytes,
                manga_id=manga_id,
                page_index=page_index,
                chapter_number=chapter_number,
                model=model,
                genre_list=genre_list,
                previous_chapter_summary=chapter_summary,
            )
            return JSONResponse(content={
                "status":            "success",
                "model_used":        result["model_used"],
                "genre_used":        genre_list,
                "has_dialogue":      result["has_dialogue"],
                "time_taken":        result["timings"]["total"],
                "timings":           result["timings"],
                "translations":      result["translations"],
                "translated_image":  result["rendered_base64"],
                "page_summary":      result["page_summary"],
                "character_updates": result["character_updates"],
                "pronoun_shift":     result["pronoun_shift"],
            })

        # Multi-page: sequential, page_index tang dan tu gia tri user nhap
        page_results = []
        for i, f in enumerate(files):
            idx        = page_index + i
            file_bytes = await f.read()
            result     = await _translate_single_page(
                file_bytes,
                manga_id=manga_id,
                page_index=idx,
                chapter_number=chapter_number,
                model=model,
                genre_list=genre_list,
                previous_chapter_summary=chapter_summary,
            )
            page_results.append({
                "page_index":      idx,
                "filename":        f.filename or f"page_{idx}",
                "translated_image": result["rendered_base64"],
                "translations":    result["translations"],
                "has_dialogue":    result["has_dialogue"],
                "time_taken":      result["timings"]["total"],
            })
            logger.info(f"[MultiPage] Done page {idx} ({i+1}/{len(files)})")

        return JSONResponse(content={
            "status":  "success",
            "total":   len(files),
            "results": page_results,
        })

    except Exception as err:
        logger.error(f"Critical error during API execution: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi he thong: {str(err)}")


# ==================================================================
# Phase 2 (shared): Sequential Render — dùng chung cho cả 2 mode
# ==================================================================

async def _run_render_phase(
    vlm_results: list[dict],
    out_dir: Path,
    chapter_number: int,
    manga_id: str,
    manga_name: str = "",
    on_page_ready=None,
) -> dict:
    """
    Render phase dùng chung cho translate_async và translate_sync.
    Nhận list VLM results (sắp xếp theo page_index), chạy LaMa + render text.
    Sequential vì LaMa là blocking PyTorch — không thể parallel hóa.
    """
    total = len(vlm_results)
    logger.info(f"[Ch{chapter_number}] Phase 2/2 start — render {total} pages (sequential)")
    t_phase2 = time.perf_counter()

    all_page_summaries:    list[str] = []
    all_character_updates: list      = []
    all_pronoun_shifts:    list      = []
    saved_paths:  list[str]  = []
    page_results: list[dict] = []

    for vlm_result in vlm_results:
        page_index = vlm_result.get("page_index", 0)

        if vlm_result.get("error"):
            logger.error(
                f"[Ch{chapter_number}|p{page_index}] VLM error — skip render: {vlm_result['error']}"
            )
            page_results.append({"page_index": page_index, "error": vlm_result["error"]})
            continue

        try:
            rendered = await _render_single_page(vlm_result)

            filename = f"page_{page_index + 1:03d}.png"
            filepath = out_dir / filename
            rendered["rendered_image"].save(str(filepath), format="PNG")
            saved_paths.append(str(filepath))

            if rendered["page_summary"]:
                all_page_summaries.append(rendered["page_summary"])
            all_character_updates.extend(rendered["character_updates"])
            all_pronoun_shifts.extend(rendered["pronoun_shift"])

            page_results.append({
                "page_index":   page_index,
                "filename":     filename,
                "preview_url":  make_library_url(manga_name, manga_id, chapter_number, filename),
                "has_dialogue": rendered["has_dialogue"],
                "page_summary": rendered["page_summary"],
                "time_taken":   rendered["timings"]["total"],
            })
            logger.debug(
                f"[Ch{chapter_number}|p{page_index}] render OK — "
                f"{rendered['timings']['total']:.1f}s -> {filename}"
            )
            if on_page_ready:
                on_page_ready(page_index, make_library_url(manga_name, manga_id, chapter_number, filename))

        except Exception as render_err:
            logger.error(
                f"[Ch{chapter_number}|p{page_index}] Render failed: {render_err}",
                exc_info=True,
            )
            page_results.append({"page_index": page_index, "error": str(render_err)})

    translated_pages = len([r for r in page_results if "error" not in r])
    err_note = f" | ⚠️ {total - translated_pages} lỗi" if translated_pages < total else ""
    logger.info(
        f"[Ch{chapter_number}] Phase 2 done — {time.perf_counter() - t_phase2:.1f}s | "
        f"{translated_pages}/{total} render OK{err_note}"
    )

    return {
        "manga_id":         manga_id,
        "chapter_number":   chapter_number,
        "total_pages":      total,
        "translated_pages": translated_pages,
        "saved_paths":      saved_paths,
        "page_results":     page_results,
        "output_dir":       str(out_dir),
        # Dữ liệu intelligence tổng hợp từ toàn chapter — dùng cho background Celery task
        "page_summaries":    all_page_summaries,
        "character_updates": all_character_updates,
        "pronoun_shifts":    all_pronoun_shifts,
    }


# ==================================================================
# translate_async — Work Queue VLM (flash-lite) + Sequential Render
# ==================================================================

_ASYNC_FLASH_MODEL = "gemini-3.1-flash-lite"
_ASYNC_MAX_RETRIES = 5   # max lần re-queue 1 trang khi bị RPM


async def _async_vlm_worker(
    work_q: asyncio.Queue,
    key_pool: asyncio.Queue,
    done_event: asyncio.Event,
    vlm_results: dict,
    cooldown: float,
    manga_id: str,
    genre_list: list[str],
    previous_chapter_summary: str,
    manga_name: str,
    chapter_number: int,
    chapter_label: str,
    model: str = "",
    pbar=None,       # tqdm progress bar (optional)
) -> None:
    """
    Worker coroutine cho translate_async Work Queue.

    Vòng lặp chính:
      1. work_q.get(timeout=2s) — chờ item (page_index, img_bytes, retry_count)
      2. key_pool.get()         — chờ key rảnh (có thể block lâu khi tất cả key đang cooldown)
      3. _vlm_single_page()     — dịch trang với key được giao (flash-lite, Queue mode)
      4. Xử lý kết quả:
         - ✅ Thành công          → lưu result, trả key cooldown bình thường
         - ❌ AllModelsRPMError   → re-queue trang (retry+1), key cooldown ×2.5
         - ❌ KeyFullyExhaustedError → re-queue, key delay đến daily reset
         - ❌ Lỗi khác / max retry → mark error, không re-queue
      5. work_q.task_done()     — luôn gọi (kể cả khi re-queue)
         Re-queue tạo TASK MỚI (+1 unfinished), task_done đóng TASK CŨ (-1).
         Net effect: unfinished count giữ nguyên khi re-queue, giảm khi success.

    Worker thoát khi done_event được set (work_q.join() đã return).
    """
    while not done_event.is_set():
        try:
            item = await asyncio.wait_for(work_q.get(), timeout=2.0)
        except asyncio.TimeoutError:
            continue   # check done_event rồi loop lại

        page_index, img_bytes, retry_count = item
        key = await key_pool.get()   # block đến khi có key rảnh

        try:
            result = await _vlm_single_page(
                img_bytes,
                manga_id=manga_id,
                page_index=page_index,
                model=model or _ASYNC_FLASH_MODEL,
                genre_list=genre_list,
                previous_chapter_summary=previous_chapter_summary,
                preferred_key=key,
                single_model_only=True,  # Không waste calls vào model fallback
                manga_name=manga_name,
                chapter_number=chapter_number,
            )
            vlm_results[page_index] = result
            if pbar is not None:
                pbar.update(1)
            schedule_key_return(key_pool, key, cooldown)
            logger.debug(
                f"[{chapter_label}|p{page_index}] ✅ "
                f"(retry={retry_count}, key=...{key.key[-4:]})"
            )

        except AllModelsRPMError:
            extra_cd = round(cooldown * 2.5, 1)
            schedule_key_return(key_pool, key, extra_cd)
            if retry_count < _ASYNC_MAX_RETRIES:
                logger.warning(
                    f"[{chapter_label}|p{page_index}] RPM → re-queue "
                    f"(attempt {retry_count + 1}/{_ASYNC_MAX_RETRIES}), "
                    f"key[...{key.key[-4:]}] cooldown={extra_cd}s"
                )
                work_q.put_nowait((page_index, img_bytes, retry_count + 1))
            else:
                logger.error(
                    f"[{chapter_label}|p{page_index}] Max retry ({_ASYNC_MAX_RETRIES}) — giving up."
                )
                vlm_results[page_index] = make_vlm_error(page_index, "rpm_max_retries")
                if pbar is not None:
                    pbar.update(1)

        except KeyFullyExhaustedError:
            secs = seconds_until_reset(key)
            schedule_key_return(key_pool, key, secs)
            if retry_count < _ASYNC_MAX_RETRIES:
                logger.warning(
                    f"[{chapter_label}|p{page_index}] Key RPD exhausted → "
                    f"delay {secs / 3600:.1f}h, re-queue (attempt {retry_count + 1})"
                )
                work_q.put_nowait((page_index, img_bytes, retry_count + 1))
            else:
                vlm_results[page_index] = make_vlm_error(page_index, "rpd_max_retries")
                if pbar is not None:
                    pbar.update(1)

        except Exception as err:
            logger.error(
                f"[{chapter_label}|p{page_index}] Unexpected error: {err}",
                exc_info=False,
            )
            schedule_key_return(key_pool, key, cooldown)
            vlm_results[page_index] = make_vlm_error(page_index, str(err))
            if pbar is not None:
                pbar.update(1)

        finally:
            work_q.task_done()   # Luôn gọi — kể cả khi đã re-queue


async def translate_async(
    images_bytes: list[bytes],
    *,
    manga_id: str,
    manga_name: str,
    chapter_number: int,
    genre_list: list[str] | None = None,
    save_dir: Path | None = None,
    on_page_ready=None,
) -> dict:
    """
    Dịch chapter: VLM song song (Work Queue) + Render tuần tự.

    Phase 1 VLM Work Queue:
      Model cố định: gemini-3.1-flash-lite.
      N worker coroutines (N = số key) chạy đồng thời.
      Khi trang bị RPM re-queue với retry_count+1, key nghỉ extra cooldown (2.5×).
      Tối đa _ASYNC_MAX_RETRIES lần retry mỗi trang trước khi mark error.

    Phase 2 Render (sequential, dùng _run_render_phase).
    """
    genre_list    = genre_list or []
    cooldown      = get_model_limit(_ASYNC_FLASH_MODEL)["cooldown_seconds"]
    out_dir       = save_dir or make_library_path(manga_name, manga_id, chapter_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    total         = len(images_bytes)
    key_pool      = get_key_pool()
    chapter_label = f"Async Ch{chapter_number}"

    reset_key_stats()   # Reset counters trước chapter mới

    logger.info(
        f"[{chapter_label}] START — manga={manga_id!r} | {total}p | "
        f"model={_ASYNC_FLASH_MODEL!r} | cooldown={cooldown}s | "
        f"pool={key_pool.qsize()} keys | max_retry={_ASYNC_MAX_RETRIES}"
    )

    previous_chapter_summary = await fetch_previous_chapter_summary(manga_id, chapter_number)
    if previous_chapter_summary:
        logger.info(f"[{chapter_label}] Prev-chapter summary loaded ({len(previous_chapter_summary)} chars)")

    # Phase 1: Work Queue VLM
    work_q: asyncio.Queue        = asyncio.Queue()
    vlm_results: dict[int, dict] = {}
    done_event = asyncio.Event()

    for i, img in enumerate(images_bytes):
        work_q.put_nowait((i, img, 0))   # (page_index, img_bytes, retry_count=0)

    from tqdm import tqdm
    pbar = tqdm(total=total, desc=f"[VLM] {chapter_label}", unit="p", ncols=90, leave=True)

    n_workers    = 5 #max(1, key_pool.qsize())
    worker_tasks = [
        asyncio.create_task(
            _async_vlm_worker(
                work_q, key_pool, done_event, vlm_results,
                cooldown, manga_id, genre_list, previous_chapter_summary, manga_name, chapter_number, chapter_label,
                model=_ASYNC_FLASH_MODEL,
                pbar=pbar,
            )
        )
        for _ in range(n_workers)
    ]

    logger.info(f"[{chapter_label}] Phase 1/2 start {n_workers} workers | {total} pages queued")
    t_phase1 = time.perf_counter()

    await work_q.join()    # Block đến khi tất cả page done (kể cả retry)
    done_event.set()       # Signal workers thoát vòng lặp
    await asyncio.gather(*worker_tasks, return_exceptions=True)  # Cleanup
    pbar.close()

    n_ok  = sum(1 for r in vlm_results.values() if not r.get("error"))
    n_err = total - n_ok
    t1    = time.perf_counter() - t_phase1
    logger.info(
        f"[{chapter_label}] Phase 1 done — {t1:.1f}s | {n_ok}/{total} OK"
        + (f" | ⚠️ {n_err} trang lỗi" if n_err else "")
    )
    log_key_stats(chapter_label)

    vlm_list = [vlm_results.get(i, make_vlm_error(i, "worker_missed")) for i in range(total)]

    # Phase 2: Render
    result = await _run_render_phase(vlm_list, out_dir, chapter_number, manga_id, manga_name, on_page_ready)
    result["manga_name"] = manga_name
    return result


# Alias dùng bởi scrape.py (backward compat)
translate_chapter = translate_async


# ==================================================================
# Rich Mode — Chế độ người giàu: chỉ Key #1 (Tier 1), 20 concurrent
# ==================================================================

_RICH_CONCURRENT = 30    # Số task chạy đồng thời tối đa cho Rich Mode
_RICH_COOLDOWN   = 3.0   # Giây chờ khi bị RPM (tăng dần theo attempt)


async def translate_rich(
    images_bytes: list[bytes],
    *,
    manga_id: str,
    manga_name: str,
    chapter_number: int,
    model: str = "gemini-3.1-flash-lite",
    genre_list: list[str] | None = None,
    save_dir: Path | None = None,
    on_page_ready=None,
) -> dict:
    """
    Chế độ Người Giàu: 20 tasks song song, chỉ dùng Key #1 (Tier 1).

    Thiết kế đơn giản — không Queue, không Pool:
      - asyncio.Semaphore(20) giới hạn số task đồng thời.
      - Mỗi task dùng trực tiếp tier1_key (Key object duy nhất).
      - Khi bị RPM: asyncio.sleep(cooldown × attempt) rồi retry trong task.
      - single_model_only=True: không fallback sang model khác.
    """
    genre_list    = genre_list or []
    out_dir       = save_dir or make_library_path(manga_name, manga_id, chapter_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    total         = len(images_bytes)
    chapter_label = f"Rich Ch{chapter_number}"

    # Lấy key từ Pro Key Pool (GEMINI_PRO_API_KEYS)
    pro_pool  = get_key_pool("pro")
    tier1_key = await pro_pool.get()

    reset_key_stats()
    logger.info(
        f"[{chapter_label}] START — manga={manga_id!r} | {total}p | "
        f"model={model!r} | concurrent={_RICH_CONCURRENT} | "
        f"key=Pro[...{tier1_key.key[-4:]}]"
    )

    previous_chapter_summary = await fetch_previous_chapter_summary(manga_id, chapter_number)
    if previous_chapter_summary:
        logger.info(f"[{chapter_label}] Prev summary: {len(previous_chapter_summary)} chars")

    # ── Phase 1: VLM song song (Semaphore) ─────────────────────────────────
    from tqdm import tqdm
    pbar = tqdm(total=total, desc=f"[VLM] {chapter_label}", unit="p", ncols=90, leave=True)
    sem  = asyncio.Semaphore(_RICH_CONCURRENT)

    async def _translate_one(page_index: int, img_bytes: bytes) -> dict:
        async with sem:
            for attempt in range(1, _ASYNC_MAX_RETRIES + 2):
                try:
                    result = await _vlm_single_page(
                        img_bytes,
                        manga_id=manga_id,
                        page_index=page_index,
                        model=model,
                        genre_list=genre_list,
                        previous_chapter_summary=previous_chapter_summary,
                        preferred_key=tier1_key,
                        single_model_only=True,
                        manga_name=manga_name,
                        chapter_number=chapter_number,
                    )
                    pbar.update(1)
                    return result
                except AllModelsRPMError:
                    wait = _RICH_COOLDOWN * attempt
                    if attempt <= _ASYNC_MAX_RETRIES:
                        logger.warning(
                            f"[{chapter_label}|p{page_index}] RPM → sleep {wait:.1f}s "
                            f"(attempt {attempt}/{_ASYNC_MAX_RETRIES})"
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"[{chapter_label}|p{page_index}] Max retry — giving up.")
                        pbar.update(1)
                        return make_vlm_error(page_index, "rpm_max_retries")
                except Exception as err:
                    logger.error(f"[{chapter_label}|p{page_index}] Error: {err}", exc_info=False)
                    pbar.update(1)
                    return make_vlm_error(page_index, str(err))

    logger.info(f"[{chapter_label}] Phase 1/2 start — {total} pages | max {_RICH_CONCURRENT} concurrent")
    t_phase1 = time.perf_counter()

    try:
        vlm_results: list[dict] = list(await asyncio.gather(
            *(_translate_one(i, img) for i, img in enumerate(images_bytes))
        ))

        n_ok = sum(1 for r in vlm_results if not r.get("error"))
        t1   = time.perf_counter() - t_phase1
        logger.info(
            f"[{chapter_label}] Phase 1 done — {t1:.1f}s | {n_ok}/{total} OK"
            + (f" | ⚠️ {total - n_ok} lỗi" if n_ok < total else "")
        )
        log_key_stats(chapter_label)

        # ── Phase 2: Render ──────────────────────────────────────────────────────
        result = await _run_render_phase(vlm_results, out_dir, chapter_number, manga_id, manga_name, on_page_ready)
        result["manga_name"] = manga_name
        return result
    finally:
        pbar.close()
        # Trả pro key về pool sau khi chapter xong (dù thành công hay lỗi)
        pro_pool.put_nowait(tier1_key)

# ==================================================================
# translate_sync — Sequential VLM (full model fallback) + Sequential Render
# ==================================================================

async def translate_sync(
    images_bytes: list[bytes],
    *,
    manga_id: str,
    manga_name: str,
    chapter_number: int,
    model: str = "gemini-3.1-flash-lite",
    genre_list: list[str] | None = None,
    save_dir: Path | None = None,
    on_page_ready=None,
) -> dict:
    """
    Dịch chapter tuần tự: 1 trang VLM → tiếp theo → ... → Render all.

    Phase 1 — VLM Sequential (Queue mode, free-tier keys only):
      Mỗi trang acquire 1 key từ get_key_pool() (chỉ chứa free keys, không có Key #1 Tier1).
      Sau khi xong trả key về pool với cooldown — cùng cơ chế với translate_async.
      Chậm hơn translate_async nhưng không bao giờ burst, độ tin cậy cao hơn.

    Phase 2 — Render (sequential, dùng _run_render_phase).
    """
    genre_list    = genre_list or []
    out_dir       = save_dir or make_library_path(manga_name, manga_id, chapter_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    total         = len(images_bytes)
    chapter_label = f"Sync Ch{chapter_number}"

    reset_key_stats()

    logger.info(
        f"[{chapter_label}] START — manga={manga_id!r} | {total}p | "
        f"model={model!r} | mode=sequential (full model fallback)"
    )

    previous_chapter_summary = await fetch_previous_chapter_summary(manga_id, chapter_number)
    if previous_chapter_summary:
        logger.info(f"[{chapter_label}] Prev-chapter summary loaded ({len(previous_chapter_summary)} chars)")

    # Phase 1: Sequential VLM
    logger.info(f"[{chapter_label}] Phase 1/2 start — sequential VLM x{total} pages")
    t_phase1 = time.perf_counter()
    vlm_results: list[dict] = []

    from tqdm import tqdm
    pool = get_key_pool()
    with tqdm(total=total, desc=f"[VLM] {chapter_label}", unit="p", ncols=90, leave=True) as pbar:
        for i, img_bytes in enumerate(images_bytes):
            key = await pool.get()
            try:
                result = await _vlm_single_page(
                    img_bytes,
                    manga_id=manga_id,
                    page_index=i,
                    model=model,
                    genre_list=genre_list,
                    previous_chapter_summary=previous_chapter_summary,
                    preferred_key=key,
                    manga_name=manga_name,
                    chapter_number=chapter_number,
                )
                vlm_results.append(result)
                logger.debug(f"[{chapter_label}|p{i}] sync OK")
            except Exception as err:
                logger.error(f"[{chapter_label}|p{i}] VLM failed: {err}", exc_info=False)
                vlm_results.append(make_vlm_error(i, str(err)))
            finally:
                cooldown = get_model_limit(model).get("cooldown_seconds", 10)
                schedule_key_return(pool, key, cooldown)
                pbar.update(1)

    n_ok = sum(1 for r in vlm_results if not r.get("error"))
    t1   = time.perf_counter() - t_phase1
    logger.info(
        f"[{chapter_label}] Phase 1 done — {t1:.1f}s | {n_ok}/{total} OK"
        + (f" | ⚠️ {total - n_ok} lỗi" if n_ok < total else "")
    )
    log_key_stats(chapter_label)

    # Phase 2: Render
    result = await _run_render_phase(vlm_results, out_dir, chapter_number, manga_id, manga_name, on_page_ready)
    result["manga_name"] = manga_name
    return result


# ==================================================================
# translate_novel — Text-only pipeline (web novel / light novel)
# ==================================================================
# Không có render phase, không có bounding box, không có ảnh.
# Input:  list[str] các đoạn văn (paragraphs) của chapter.
# Output: {translated_text, chapter_summary, character_updates, pronoun_shifts}
#
# Reuses:
#   - fetch_previous_chapter_summary()  — lấy context chapter trước
#   - GraphService.load()               — lấy character graph
#   - KeyManager                        — chọn API key
#   - GeminiModel.translate_text()      — text-only Gemini call
#
# Không cần pool (asyncio.Queue) vì novel chỉ gọi 1-2 requests/chapter.
# RPM retry: thử lần lượt qua free keys (tối đa MAX_KEY_RETRIES key).
# ==================================================================

_NOVEL_MAX_KEY_RETRIES = 3   # Thử tối đa N free keys trước khi raise
_NOVEL_RPM_BACKOFF     = 5.0 # Giây chờ khi bị RPM trước khi thử key tiếp theo


async def translate_novel(
    paragraphs: list[str],
    *,
    manga_id: str,
    manga_name: str,
    chapter_number: int,
    model: str = "gemini-3.1-flash-lite",
    genre_list: list[str] | None = None,
) -> dict:
    """
    Dịch một chapter novel (plain text) sang tiếng Việt trong 1 API call.

    Parameters
    ----------
    paragraphs:      Danh sách đoạn văn cần dịch (từ extension scrape).
    manga_id:        ID của bản ghi novel trong DB (có thể kết thúc bằng _novel slug).
    manga_name:      Tên hiển thị (đã có " [Novel]" suffix nếu cần phân biệt).
    chapter_number:  Số chapter.
    model:           Gemini model để dịch.
    genre_list:      Thể loại để nạp skill vào prompt.

    Returns
    -------
    dict with keys:
        translated_text   — full chapter text đã dịch
        chapter_summary   — tóm tắt ngắn (dùng làm context chapter tiếp)
        character_updates — list nhân vật xuất hiện (format giống manga)
        pronoun_shifts    — list thay đổi xưng hô (format giống manga)
    """
    from app.ai.gemini import GeminiModel
    from app.ai.base_model import BaseModel
    from app.ai.key_manager import KeyManager
    from app.prompts.novel_prompt import NOVEL_RESPONSE_SCHEMA, _NOVEL_SYSTEM_PROMPT_TEMPLATE

    genre_list    = genre_list or []
    chapter_label = f"Novel Ch{chapter_number}"

    logger.info(
        f"[{chapter_label}] START — novel={manga_id!r} | {len(paragraphs)} paragraphs "
        f"| model={model!r} | genres={genre_list}"
    )
    t0 = time.perf_counter()

    # ── Context: chapter trước + character graph (tái sử dụng hoàn toàn từ manga) ──
    previous_summary  = await fetch_previous_chapter_summary(manga_id, chapter_number)
    G                 = await GraphService.load(manga_id, use_cache=True)
    character_graph   = GraphService.to_prompt(G)

    if previous_summary:
        logger.info(f"[{chapter_label}] Prev summary loaded ({len(previous_summary)} chars)")

    # ── Gom toàn bộ paragraphs thành 1 text block ──
    # Dùng dấu phân cách rõ ràng để model biết ranh giới paragraph khi dịch.
    full_text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    if not full_text:
        raise ValueError("Novel content trống — không có gì để dịch.")

    logger.info(f"[{chapter_label}] Text length: {len(full_text)} chars")

    # ── Build system prompt ──
    system_prompt = BaseModel.build_system_prompt(
        _NOVEL_SYSTEM_PROMPT_TEMPLATE,
        genre=genre_list,
        character_graph=character_graph,
        previous_chapter_summary=previous_summary,
        manga_name=manga_name,
        chapter_number=chapter_number,
    )

    # ── Chọn key + retry đơn giản (không dùng Pool vì chỉ 1-2 calls) ──
    km        = KeyManager()
    free_keys = [k for k in km.keys if k.tier == "free"]
    if not free_keys:
        raise RuntimeError("Không có free API key khả dụng cho novel translation.")

    raw_response: str | None = None
    last_err: Exception | None = None

    for attempt, key in enumerate(free_keys[:_NOVEL_MAX_KEY_RETRIES], start=1):
        try:
            model_obj    = GeminiModel(model_name=model, api_key=key.key)
            raw_response = await model_obj.translate_text(full_text, system_prompt, NOVEL_RESPONSE_SCHEMA)
            logger.debug(f"[{chapter_label}] API call OK (key #{attempt})")
            break
        except Exception as err:
            last_err = err
            is_rpm   = "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err)
            if is_rpm and attempt < min(_NOVEL_MAX_KEY_RETRIES, len(free_keys)):
                wait = _NOVEL_RPM_BACKOFF * attempt
                logger.warning(f"[{chapter_label}] RPM key #{attempt} — wait {wait:.0f}s, try next key")
                await asyncio.sleep(wait)
            else:
                logger.error(f"[{chapter_label}] API failed key #{attempt}: {err}")
                if attempt >= min(_NOVEL_MAX_KEY_RETRIES, len(free_keys)):
                    break

    if raw_response is None:
        raise RuntimeError(f"[{chapter_label}] Tất cả keys đều thất bại: {last_err}")

    # ── Parse JSON response ──
    import json
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as e:
        # Fallback: thử dùng extract_json_block nếu có prefix/suffix thừa
        from app.utils.json_parser import extract_json_block
        try:
            data = json.loads(extract_json_block(raw_response))
        except Exception:
            logger.error(f"[{chapter_label}] Cannot parse novel response: {e}\nRaw: {raw_response[:200]}")
            raise RuntimeError(f"Novel API response không parse được: {e}")

    elapsed = time.perf_counter() - t0
    translated = data.get("translated_text", "")
    logger.info(
        f"[{chapter_label}] DONE — {elapsed:.1f}s | "
        f"{len(translated)} chars translated | "
        f"{len(data.get('character_updates', []))} char_updates"
    )

    return {
        "translated_text":   translated,
        "chapter_summary":   data.get("chapter_summary", ""),
        "character_updates": data.get("character_updates", []),
        "pronoun_shifts":    data.get("pronoun_shifts", []),
    }
