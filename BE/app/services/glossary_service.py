"""
Module: app.services.glossary_service
Description: Business logic cho Glossary Engine:
  - Load glossary terms cho System Prompt
  - Post-chapter aggregator: gom, chuẩn hóa, deduplicate, và lưu vào DB
"""

import logging
from app.services.db_service import AsyncSessionLocal, GlossaryCRUD

logger = logging.getLogger(__name__)


async def load_glossary_for_manga(manga_id: str) -> list[tuple[str, str]]:
    """
    Load danh sách thuật ngữ từ DB cho một bộ truyện.
    Trả về list[(source_term, target_term)] để nạp vào System Prompt.
    """
    try:
        async with AsyncSessionLocal() as session:
            terms = await GlossaryCRUD.get_by_manga_id(session, manga_id)
            glossary = [(t.source_term, t.target_term) for t in terms]
            if glossary:
                logger.info(f"[Glossary] Loaded {len(glossary)} term(s) for manga={manga_id!r}")
            return glossary
    except Exception as err:
        logger.warning(f"[Glossary] Failed to load for manga={manga_id!r}: {err}")
        return []


async def aggregate_and_save_glossary(
    manga_id: str,
    vlm_results: list[dict],
    existing_glossary: list[tuple[str, str]],
) -> int:
    """
    Post-Chapter Aggregator: gom tất cả new_terms_discovered từ các trang,
    chuẩn hóa, gộp nối và lưu vào DB.

    - Gom nhóm theo source_term.strip().lower()
    - Nối các target_term khác nhau bằng dấu phẩy
    - Bỏ qua từ đã có trong DB (existing_glossary)

    Returns: Số thuật ngữ mới được lưu vào DB.
    """
    # Collect all new_terms from VLM results
    raw_terms: list[dict] = []
    for result in vlm_results:
        if result.get("error"):
            continue
        vlm_resp = result.get("vlm_response", {})
        new_terms = vlm_resp.get("new_terms_discovered", [])
        if isinstance(new_terms, list):
            raw_terms.extend(new_terms)

    if not raw_terms:
        return 0

    # Build set of existing source_terms (lowered) for fast lookup
    existing_sources = {src.strip().lower() for src, _ in existing_glossary}

    # Group by normalized source_term
    grouped: dict[str, list[str]] = {}
    display_source: dict[str, str] = {}  # Keep the nicest display form
    for entry in raw_terms:
        source = entry.get("source_term", "").strip()
        target = entry.get("target_term", "").strip()
        if not source or not target:
            continue
        key = source.lower()
        # Skip if already in DB
        if key in existing_sources:
            continue
        if key not in grouped:
            grouped[key] = []
            display_source[key] = source  # Keep first Title Case version
        if target not in grouped[key]:
            grouped[key].append(target)

    if not grouped:
        return 0

    # Build final dict: source_term -> "target1, target2"
    new_terms_dict: dict[str, str] = {}
    for key, targets in grouped.items():
        source = display_source[key]
        new_terms_dict[source] = ", ".join(targets)

    # Save to DB
    try:
        async with AsyncSessionLocal() as session:
            added = await GlossaryCRUD.save_chapter_terms(session, manga_id, new_terms_dict)
            if added > 0:
                terms_preview = "; ".join(
                    f'"{s}" → "{t}"' for s, t in list(new_terms_dict.items())[:5]
                )
                logger.info(
                    f"[Glossary] Saved {added} new term(s) for manga={manga_id!r}: {terms_preview}"
                )
            return added
    except Exception as err:
        logger.warning(f"[Glossary] Failed to save terms for manga={manga_id!r}: {err}")
        return 0
