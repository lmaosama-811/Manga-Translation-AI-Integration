#!/usr/bin/env python3
"""
test_pipeline.py — Standalone pipeline test cho Manga-Translation-AI-Integration.

Cách dùng:
  # Dịch 1 ảnh:
  python test_pipeline.py /path/to/page.jpg

  # Dịch nhiều ảnh:
  python test_pipeline.py /path/to/page1.jpg /path/to/page2.png

  # Tùy chọn thêm:
  python test_pipeline.py page.jpg --model gemini-3.1-flash-lite --genre action,comedy --manga-id my-manga

Đầu ra:
  - Ảnh đã dịch lưu vào thư mục cùng tên với file ảnh gốc (bỏ phần mở rộng).
    Ví dụ: /path/to/page.jpg → /path/to/page/page_001.png
"""

import sys
import os
import asyncio
import argparse
import time
import logging
from pathlib import Path

# ── Đảm bảo chạy đúng sys.path từ thư mục BE/ ──────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_BE_DIR = _SCRIPT_DIR / "BE"

if str(_BE_DIR) not in sys.path:
    sys.path.insert(0, str(_BE_DIR))

# Chạy từ thư mục BE/ để config.py tìm đúng .env
os.chdir(_BE_DIR)

# ── Setup logging trước khi import app ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger("test_pipeline")


# ── Hàm test chính ──────────────────────────────────────────────────────────

async def test_single_image(
    image_path: Path,
    output_dir: Path,
    manga_id: str,
    model: str,
    genre_list: list,
    page_index: int = 0,
) -> dict:
    """
    Dịch 1 ảnh manga, lưu kết quả vào output_dir.
    Gọi trực tiếp vào _translate_single_page (giống Playground endpoint).
    """
    from app.api.routes.translate import _translate_single_page

    logger.info(f"[Page {page_index}] Bắt đầu dịch: {image_path.name}")
    t_start = time.perf_counter()

    file_bytes = image_path.read_bytes()

    result = await _translate_single_page(
        file_bytes,
        manga_id=manga_id,
        page_index=page_index,
        model=model,
        genre_list=genre_list,
        manga_name=image_path.stem,
    )

    elapsed = time.perf_counter() - t_start

    # ── Lưu ảnh đầu ra ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"page_{page_index + 1:03d}.png"
    output_path = output_dir / output_filename

    rendered_image = result.get("rendered_image")
    if rendered_image:
        rendered_image.save(str(output_path), format="PNG")
        logger.info(f"[Page {page_index}] Đã lưu: {output_path}")
    else:
        logger.warning(f"[Page {page_index}] Không có rendered_image trong kết quả!")

    # ── In kết quả tóm tắt ──────────────────────────────────────────────────
    timings    = result.get("timings", {})
    has_dial   = result.get("has_dialogue", False)
    model_used = result.get("model_used", model)
    num_bubbles = len(result.get("translations", []))

    print(f"\n{'─'*60}")
    print(f"  File:         {image_path.name}")
    print(f"  Model:        {model_used}")
    print(f"  Has dialogue: {has_dial}  |  Bubbles dịch: {num_bubbles}")
    print(f"  Timings:")
    print(f"    • Image prep:   {timings.get('image_compression', 0):.2f}s")
    print(f"    • VLM API:      {timings.get('vlm_api_call', 0):.2f}s")
    if has_dial:
        print(f"    • Typesetting:  {timings.get('bubble_analysis_typesetting', 0):.2f}s")
        print(f"    • Inpainting:   {timings.get('inpainting', 0):.2f}s")
        print(f"    • Rendering:    {timings.get('rendering', 0):.2f}s")
    print(f"    • Total:        {elapsed:.2f}s")
    if output_path.exists():
        print(f"  Output:       {output_path}")
    print(f"{'─'*60}\n")

    # ── In chi tiết các bong bóng dịch được ─────────────────────────────────
    translations = result.get("translations", [])
    if translations:
        print("  Bản dịch từng bong bóng:")
        for i, t in enumerate(translations, 1):
            text_vi = t.get("text_vi", "")
            bid     = t.get("bubble_id", i)
            print(f"    [{bid}] {text_vi}")
        print()

    return {
        "page_index":   page_index,
        "output_path":  str(output_path),
        "has_dialogue": has_dial,
        "num_bubbles":  num_bubbles,
        "elapsed":      elapsed,
        "error":        None,
    }


async def run_test(args):
    """Chạy test cho một hoặc nhiều ảnh."""
    image_paths = [Path(p).resolve() for p in args.images]

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in image_paths:
        if not p.exists():
            logger.error(f"File không tồn tại: {p}")
            sys.exit(1)
        if p.suffix.lower() not in valid_exts:
            logger.error(f"Định dạng không hỗ trợ: {p.suffix} (chỉ hỗ trợ JPG, PNG, WEBP)")
            sys.exit(1)

    # Thư mục output = thư mục cha / tên file đầu tiên (bỏ phần mở rộng)
    output_dir = image_paths[0].parent / image_paths[0].stem

    genre_list = [g.strip() for g in args.genre.split(",") if g.strip()] if args.genre else []
    manga_id   = args.manga_id or image_paths[0].stem.replace(" ", "_").lower()

    print(f"\n{'='*60}")
    print(f"  TEST PIPELINE — Manga Translation")
    print(f"{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Genre:      {genre_list or '(none)'}")
    print(f"  Manga ID:   {manga_id}")
    print(f"  Pages:      {len(image_paths)}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")

    t_total_start = time.perf_counter()
    results = []

    for i, img_path in enumerate(image_paths):
        try:
            res = await test_single_image(
                image_path=img_path,
                output_dir=output_dir,
                manga_id=manga_id,
                model=args.model,
                genre_list=genre_list,
                page_index=i,
            )
            results.append(res)
        except KeyboardInterrupt:
            logger.info("Bị ngắt bởi người dùng.")
            break
        except Exception as e:
            logger.error(f"Lỗi khi dịch {img_path.name}: {e}", exc_info=True)
            results.append({
                "page_index":   i,
                "output_path":  None,
                "has_dialogue": False,
                "num_bubbles":  0,
                "elapsed":      0.0,
                "error":        str(e),
            })

    t_total   = time.perf_counter() - t_total_start
    ok_count  = sum(1 for r in results if not r["error"])
    err_count = len(results) - ok_count

    print(f"\n{'='*60}")
    print(f"  TỔNG KẾT")
    print(f"{'='*60}")
    print(f"  Thành công:     {ok_count}/{len(results)} trang")
    if err_count:
        print(f"  Lỗi:            {err_count} trang")
    print(f"  Tổng thời gian: {t_total:.1f}s")
    print(f"  Output dir:     {output_dir}")
    print(f"{'='*60}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test pipeline dịch manga — nhận path ảnh, lưu output vào thư mục cùng tên.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python test_pipeline.py page01.jpg
  python test_pipeline.py page01.jpg page02.jpg page03.jpg
  python test_pipeline.py page01.jpg --model gemini-3.1-flash-lite --genre action
  python test_pipeline.py page01.jpg --manga-id my-manga-slug
""",
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="Đường dẫn đến file ảnh (JPG, PNG, WEBP). Có thể truyền nhiều file.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite",
        metavar="MODEL",
        help="Tên model VLM (mặc định: gemini-3.1-flash-lite)",
    )
    parser.add_argument(
        "--genre",
        default="",
        metavar="GENRES",
        help="Thể loại manga, phân tách bằng dấu phẩy. Ví dụ: action,comedy",
    )
    parser.add_argument(
        "--manga-id",
        default="",
        metavar="ID",
        help="Slug ID bộ truyện (tự sinh nếu bỏ trống)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Bật log DEBUG chi tiết",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("app").setLevel(logging.DEBUG)

    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
