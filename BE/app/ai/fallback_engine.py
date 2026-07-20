"""
Module: app.ai.fallback_engine
Description: Hệ thống fallback 2 tầng cho VLM inference.

┌─────────────────────────────────────────────────────────────────────────┐
│  KIẾN TRÚC 2 TẦNG                                                       │
│                                                                         │
│  Tầng 2 ─ Key Retry Loop (translate._throttled_vlm)                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  - Quản lý Key Pool (asyncio.Queue)                               │  │
│  │  - await pool.get() → nhận key độc quyền                         │  │
│  │  - Gọi Tầng 1 với key đó                                         │  │
│  │  - Bắt exception từ Tầng 1 → quyết định cooldown trả key        │  │
│  │  - Retry tối đa MAX_KEY_RETRIES lần với key mới từ Queue          │  │
│  └───────────────────────┬───────────────────────────────────────────┘  │
│                           │ preferred_key (from Queue)                   │
│  Tầng 1 ─ execute_fallback_chain (file này)                             │
│  ┌───────────────────────▼───────────────────────────────────────────┐  │
│  │  - Chỉ làm việc trên 1 key được giao                             │  │
│  │  - Thử: model_primary → model_fallback_1 → model_fallback_2      │  │
│  │  - RPM 429  → đưa model về cuối queue nội bộ, thử model kế        │  │
│  │  - RPD daily → mark exhausted, skip model                         │  │
│  │  - Hết model → raise AllModelsRPMError / KeyFullyExhaustedError   │  │
│  │  - KHÔNG tự rotation key sang key khác                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Legacy mode (preferred_key=None, dùng cho Playground single-request):  │
│  - Round-robin key selection + internal key rotation như cũ             │
│  - Tầng 2 không cần thiết cho single request                            │
└─────────────────────────────────────────────────────────────────────────┘

Lợi ích so với thiết kế cũ:
  ✅ Queue luôn là source of truth duy nhất cho key rotation
  ✅ Không bao giờ bypass Queue (không còn same-key collision)
  ✅ Cooldown có thể điều chỉnh theo loại lỗi (RPM vs RPD vs thành công)
  ✅ Retry logic rõ ràng, có bound (MAX_KEY_RETRIES)
"""

import asyncio
import logging
from typing import Any, Optional

# pyrefly: ignore [missing-import]
from fastapi import HTTPException

from app.core.config import settings
from app.ai import get_vlm_model, get_fallback_model_names
from app.ai.base_model import BaseModel
from app.utils.json_parser import parse_vlm_json
from app.ai.key_manager import ErrorType, KeyManager, Key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions — Tầng 1 raise, Tầng 2 bắt
# ---------------------------------------------------------------------------

class AllModelsRPMError(Exception):
    """
    Tất cả model trên key hiện tại đều bị 429 RPM.

    Ý nghĩa: Key cần nghỉ để RPM window (60s) phục hồi.
    Hành động của Tầng 2: trả key về pool với extra cooldown (2.5× bình thường),
    lấy key mới từ Queue để retry.
    """
    pass


class KeyFullyExhaustedError(Exception):
    """
    Tất cả model trên key hiện tại đã hết quota ngày (RPD).

    Ý nghĩa: Key không dùng được cho đến khi reset hàng ngày.
    Hành động của Tầng 2: trả key về pool với delay = seconds_until_daily_reset,
    lấy key mới từ Queue để retry.
    """
    pass


# ---------------------------------------------------------------------------
# Error Classification
# ---------------------------------------------------------------------------

def classify_error(error: Exception) -> ErrorType:
    """
    Phân loại lỗi thành 3 nhóm để quyết định hành vi fallback.

    - DAILY_EXHAUSTED: 429 + "daily" / "per day" → hết quota ngày
    - RATE_LIMIT_RPM:  429 / 503 / "quota"       → vượt RPM/TPM (tạm thời)
    - PER_MODEL:       mọi thứ còn lại            → lỗi cá nhân model
    """
    error_str = str(error).lower()
    status_code = getattr(error, "status_code", 0)

    # Ưu tiên kiểm tra daily quota trước (cũng là 429 nhưng cần đánh dấu riêng)
    if any(keyword in error_str for keyword in ["daily", "per day", "daily_limit"]):
        return ErrorType.DAILY_EXHAUSTED

    # Rate limit RPM/TPM — có thể giải quyết bằng đợi hoặc đổi key
    if status_code in (429, 503) or any(keyword in error_str for keyword in [
        "resource_exhausted", "rate limit", "quota",
        "too many requests", "429", "503", "service unavailable"
    ]):
        return ErrorType.RATE_LIMIT_RPM

    # Mọi thứ khác: lỗi cá nhân model (400, 500, timeout, JSON parse...)
    return ErrorType.PER_MODEL


# ---------------------------------------------------------------------------
# Tầng 1 — Core: thử tất cả model trên 1 key duy nhất
# ---------------------------------------------------------------------------

async def _try_models_on_key(
    key: Key,
    model_names: list[str],
    vlm_image_base64: str,
    genre: list[str],
    character_graph: str,
    previous_chapter_summary: str,
    key_manager: KeyManager,
    manga_name: str = "",
    chapter_number: int = 0,
) -> tuple[dict[str, Any], BaseModel]:
    """
    Thử lần lượt tất cả model trong model_names trên 1 key duy nhất.

    Logic nội bộ:
      - PER_MODEL error   → skip model, thử model tiếp theo
      - RPM 429           → đưa model về cuối danh sách (retry 1 lần), tiếp tục
      - RPD daily         → mark exhausted trên key này, skip
      - Thành công        → return ngay (tuple result, model)

    Khi hết model:
      - Tất cả RPD        → raise KeyFullyExhaustedError
      - Có ít nhất 1 RPM  → raise AllModelsRPMError

    Args:
        key: Key object đang được giữ bởi caller (từ Queue).
        model_names: Danh sách model ưu tiên (primary trước, fallback sau).
        key_manager: Singleton KeyManager để mark exhausted.
        (các arg còn lại truyền thẳng vào VLM)

    Raises:
        AllModelsRPMError: Cần key khác, key này cần nghỉ.
        KeyFullyExhaustedError: Key hết quota ngày, loại khỏi pool.
    """
    try:
        key_label = f"Key#{key_manager.keys.index(key) + 1}"
    except ValueError:
        key_label = f"ProKey[...{key.key[-4:]}]"   # Pro key không nằm trong free key list

    # Danh sách model thử lần lượt (KHÔNG retry cùng model trên cùng key)
    # Lý do: retry gần như ngay lập tức → burst → 429 lại → vô ích
    # Tầng 2 sẽ retry với KEY MỚI thay vì model mới trên key cũ
    model_names_to_try = list(model_names)
    had_rpm_error = False              # track để quyết định exception type cuối

    for model_name in model_names_to_try:
        # Skip nếu model đã bị mark exhausted (RPD) trên key này
        if model_name in key.exhausted_models:
            continue

        vlm_model = get_vlm_model(model_name, key.key)
        try:
            logger.debug(f"[Fallback] {key_label}|{model_name} — calling VLM...")
            custom_prompt = vlm_model.build_system_prompt(
                genre=genre,
                character_graph=character_graph,
                previous_chapter_summary=previous_chapter_summary,
                manga_name=manga_name,
                chapter_number=chapter_number,
            )
            key.request_count += 1   # ← đếm request cho stats log
            response   = await vlm_model.translate(vlm_image_base64, custom_prompt)
            vlm_result = parse_vlm_json(response)

            if not isinstance(vlm_result.get("translations"), list):
                raise ValueError("VLM response missing valid 'translations' list.")
            if vlm_result.get("has_dialogue") and not vlm_result["translations"]:
                raise ValueError("has_dialogue=true but translations empty — invalid VLM response.")

            logger.debug(f"[Fallback] ✅ {key_label}|{model_name}")
            return vlm_result, vlm_model

        except Exception as err:
            error_type = classify_error(err)
            err_msg    = str(err)
            if hasattr(err, "detail"):
                err_msg = f"{err.detail} (HTTP {getattr(err, 'status_code', '?')})"

            if error_type == ErrorType.PER_MODEL:
                # Lỗi cá nhân model (400, timeout, parse...) → skip, thử tiếp
                logger.warning(f"[Fallback] {key_label}|{model_name} PER_MODEL: {err_msg[:120]} — skip model.")
                continue

            elif error_type == ErrorType.RATE_LIMIT_RPM:
                # RPM 429 — ghi nhận và chuyển sang model tiếp theo.
                # KHÔNG retry cùng model trên cùng key: burst 2 request trong 0.5s
                # sẽ trigger Google burst detection dù RPM/min vẫn trong ngưỡng.
                # Tầng 2 sẽ retry với key MỚI từ Queue.
                logger.warning(f"[Fallback] {key_label}|{model_name} RPM 429 — skip to next model (no same-key retry).")
                had_rpm_error = True
                continue   # ← chỉ next model, KHÔNG append lại

            elif error_type == ErrorType.DAILY_EXHAUSTED:
                # RPD — mark model exhausted trên key này, skip hẳn
                logger.warning(f"[Fallback] {key_label}|{model_name} DAILY_EXHAUSTED — mark & skip.")
                key_manager.mark_model_exhausted(key, model_name)
                continue

    # ── Hết model, không thành công ───────────────────────────────────────
    # Kiểm tra: tất cả model có bị RPD không?
    all_rpd = all(m in key.exhausted_models for m in model_names)

    if all_rpd:
        raise KeyFullyExhaustedError(
            f"{key_label}: all models daily-exhausted — exclude from pool until reset."
        )
    else:
        # Có ít nhất 1 model bị RPM (không phải RPD) → key cần nghỉ RPM window
        raise AllModelsRPMError(
            f"{key_label}: all models hit RPM — need fresh key."
        )


# ---------------------------------------------------------------------------
# Public API — execute_fallback_chain
# ---------------------------------------------------------------------------

async def execute_fallback_chain(
    primary_model_name: str,
    vlm_image_base64: str,
    genre: list[str],
    character_graph: str = "",
    previous_chapter_summary: str = "",
    preferred_key: Optional[Key] = None,
    single_model_only: bool = False,
    manga_name: str = "",
    chapter_number: int = 0,
) -> tuple[dict[str, Any], BaseModel]:
    """
    Điểm vào chính của hệ thống fallback (Tầng 1).

    Queue mode — preferred_key is not None:
      Chỉ thử models trên preferred_key được Queue cấp.
      Raise AllModelsRPMError hoặc KeyFullyExhaustedError → Tầng 2 xử lý.

    Legacy mode — preferred_key is None:
      Round-robin key selection + internal key rotation.

    Args:
        single_model_only: Nếu True, chỉ thử primary_model_name (không fallback sang model khác).
                           Dùng cho translate_async — khi bị RPM → re-queue ngay, không waste
                           thêm calls vào flash/2.5-flash (mỗi call thêm = thêm RPM pressure).
    """
    key_manager = KeyManager()
    # single_model_only=True: chỉ thử đúng 1 model (dùng cho async, không waste calls)
    # single_model_only=False: full fallback chain (flash-lite → flash → 2.5-flash)
    if single_model_only:
        model_names = [primary_model_name]
    else:
        model_names = [primary_model_name] + get_fallback_model_names(primary_model_name)

    # ── Queue mode ─────────────────────────────────────────────────────────
    if preferred_key is not None:
        # Tầng 1: thử models trên key được giao, raise nếu thất bại.
        # Tầng 2 (caller) sẽ bắt exception và xử lý key rotation qua Queue.
        return await _try_models_on_key(
            key                      = preferred_key,
            model_names              = model_names,
            vlm_image_base64         = vlm_image_base64,
            genre                    = genre,
            character_graph          = character_graph,
            previous_chapter_summary = previous_chapter_summary,
            key_manager              = key_manager,
            manga_name               = manga_name,
            chapter_number           = chapter_number,
        )

    # ── Legacy mode ────────────────────────────────────────────────────────
    # Hành vi cũ: round-robin key selection + internal key rotation.
    # Dùng cho Playground (single request, không có Queue).
    try:
        current_key = key_manager.get_next_key_for_models(model_names)
    except RuntimeError as all_exhausted:
        logger.error(str(all_exhausted))
        raise

    keys_tried: set[Key] = set()

    while current_key and current_key not in keys_tried:
        keys_tried.add(current_key)
        try:
            return await _try_models_on_key(
                key                      = current_key,
                model_names              = model_names,
                vlm_image_base64         = vlm_image_base64,
                genre                    = genre,
                character_graph          = character_graph,
                previous_chapter_summary = previous_chapter_summary,
                key_manager              = key_manager,
                manga_name               = manga_name,
                chapter_number           = chapter_number,
            )
        except (AllModelsRPMError, KeyFullyExhaustedError) as key_err:
            # Key hiện tại thất bại → thử key tiếp theo (legacy behavior)
            logger.warning(f"[Fallback Legacy] {key_err} — switching to next key.")
            available = key_manager.get_available_keys_for_models(model_names)
            untried   = [k for k in available if k not in keys_tried]
            if untried:
                current_key = untried[0]
                logger.warning(
                    f"[Fallback Legacy] → Key#{key_manager.keys.index(current_key) + 1}"
                )
            else:
                break   # Không còn key nào khả dụng

    logger.error("❌ [Fallback Legacy] All keys and models failed.")
    raise RuntimeError(
        "❌ All API keys and models exhausted. "
        "Chi tiết đã được log ở trên."
    )
