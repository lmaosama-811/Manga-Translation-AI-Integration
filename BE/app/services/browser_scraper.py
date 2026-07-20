"""
Module: app.services.browser_scraper
Description: Playwright Chromium browser scraping service cho mobile (/scrape/url).

== Windows Workaround ==
Playwright cần asyncio.create_subprocess_exec để spawn driver process.
Windows asyncio.SelectorEventLoop (default của uvicorn) không hỗ trợ subprocess
→ NotImplementedError.

Giải pháp: Chạy toàn bộ Playwright trong một background thread riêng
có asyncio.ProactorEventLoop của riêng nó. FastAPI giao tiếp qua
asyncio.run_coroutine_threadsafe + run_in_executor (không block main loop).

Flow:
  FastAPI (SelectorEventLoop)
    └─ scrape_with_playwright()
         └─ _run_in_pw_loop()          ← giao việc sang Playwright thread
              └─ _PlaywrightLoop       ← background thread, ProactorEventLoop
                   └─ _playwright_scrape_impl()
                        ├─ BrowserManager.get_browser()
                        ├─ page.goto(url)
                        ├─ _slow_scroll(page)
                        └─ page.evaluate(JS)
"""

import asyncio
import logging
import re
import sys
import threading
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAGE_LOAD_TIMEOUT_MS = 60_000   # 60s tổng timeout cho page.goto()
SCROLL_STEP_PX       = 250      # pixels mỗi bước scroll (nhỏ = chậm = ảnh load kịp)
SCROLL_DELAY_MS      = 300      # ms chờ giữa 2 bước scroll (mô phỏng lướt tay)
POST_SCROLL_WAIT_MS  = 3_000    # ms chờ thêm sau khi scroll xuống cuối
MAX_SCROLL_STEPS     = 300      # cap chống infinite-scroll (300 × 250px = 75,000px ≈ 80 trang)


# ---------------------------------------------------------------------------
# JS chạy bên trong Chromium — extract images + title + paragraphs
# ---------------------------------------------------------------------------

_EXTRACT_JS = """
() => {
    const AD_PATTERNS = [
        "googlesyndication", "doubleclick", "adnxs", "pubmatic",
        "taboola", "outbrain", "amazon-adsystem", "criteo",
        "facebook.com/tr", "pixel.gif", "1x1",
    ];

    function isAdUrl(url) {
        return !url || AD_PATTERNS.some(p => url.toLowerCase().includes(p));
    }

    function getBestSrc(img) {
        return img.getAttribute("data-original")
            || img.getAttribute("data-src")
            || img.getAttribute("data-lazy-src")
            || img.getAttribute("data-lazy")
            || img.currentSrc
            || img.src
            || "";
    }

    const MANGA_SELECTORS = [
        ".reading-detail img", ".reading-detail__box img",
        ".chapter-content img", ".page-chapter img",
        ".manga-reader img", "#chapter-content img",
        ".page-break img", ".container-chapter-reader img",
        ".images-content img", ".reader-content img",
        ".chapter-detail-movie img", ".nettromtrang img",
        ".wp-manga-chapter-img", ".chapter_img",
        "img[data-original]", "img[data-src]", "img[data-lazy]",
        ".viewer-img img", ".toon-img img",
    ];

    const images = [];
    const seen   = new Set();

    for (const sel of MANGA_SELECTORS) {
        try {
            const found = Array.from(document.querySelectorAll(sel));
            for (const img of found) {
                const src = getBestSrc(img);
                if (!src || !src.startsWith("http") || seen.has(src)) continue;
                if (isAdUrl(src)) continue;
                const w = img.naturalWidth  || parseInt(img.getAttribute("width")  || "0") || 0;
                const h = img.naturalHeight || parseInt(img.getAttribute("height") || "0") || 0;
                if (w > 0 && w < 100) continue;
                if (h > 0 && h < 150) continue;
                seen.add(src);
                images.push({ url: src, width: w, height: h, alt: (img.alt || "").substring(0, 100) });
            }
        } catch (_) {}
        if (images.length >= 3) break;
    }

    if (images.length === 0) {
        const allImgs = Array.from(document.querySelectorAll("img"));
        for (const img of allImgs) {
            const src = getBestSrc(img);
            if (!src || !src.startsWith("http") || seen.has(src)) continue;
            if (isAdUrl(src)) continue;
            const h = img.naturalHeight || parseInt(img.getAttribute("height") || "0") || 0;
            if (h > 0 && h < 400) continue;
            seen.add(src);
            images.push({ url: src, width: img.naturalWidth || 0, height: h, alt: (img.alt || "").substring(0, 100) });
        }
    }

    const ogMeta  = document.querySelector('meta[property="og:title"]');
    const h1El    = document.querySelector("h1");
    const titleEl = document.querySelector("title");
    const rawTitle = (ogMeta  && ogMeta.getAttribute("content"))
                  || (h1El   && h1El.textContent)
                  || (titleEl && titleEl.textContent)
                  || window.location.href;

    const NOVEL_SELS = [
        ".chapter-content p", ".box-chap p", "#chapter-content p",
        ".content-text p", ".text-content p", "article p",
    ];
    let paragraphs = [];
    for (const sel of NOVEL_SELS) {
        try {
            const ps = Array.from(document.querySelectorAll(sel))
                .map(p => p.textContent.trim()).filter(t => t.length > 20);
            if (ps.length > 0) { paragraphs = ps; break; }
        } catch (_) {}
    }

    return {
        images,
        rawTitle:  (rawTitle || "").trim().substring(0, 300),
        paragraphs,
        finalUrl:  window.location.href,
    };
}
"""


# ---------------------------------------------------------------------------
# Chapter extraction
# ---------------------------------------------------------------------------

_CHAPTER_RE = re.compile(
    r"(?:chapter|chap|ch|tap|episode)[/\-_\s\.#\(]*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def _extract_chapter(title: str, url: str) -> tuple[int, str]:
    m = _CHAPTER_RE.search(url + " " + title)
    if m:
        return int(float(m.group(1))), m.group(0)
    return 0, ""


# ---------------------------------------------------------------------------
# _PlaywrightLoop — dedicated background thread với ProactorEventLoop
#
# Windows fix: uvicorn dùng SelectorEventLoop, không hỗ trợ subprocess.
# Playwright cần subprocess → chạy trong thread riêng với ProactorEventLoop.
# ---------------------------------------------------------------------------

class _PlaywrightLoop:
    _thread: threading.Thread | None = None
    _loop:   asyncio.AbstractEventLoop | None = None
    _ready   = threading.Event()
    _lock    = threading.Lock()

    @classmethod
    def _thread_main(cls) -> None:
        """Entry point của background thread — tạo ProactorEventLoop và chạy mãi."""
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        cls._loop = loop
        asyncio.set_event_loop(loop)
        logger.info(f"[Browser] Playwright loop started ({type(loop).__name__})")
        cls._ready.set()
        loop.run_forever()

    @classmethod
    def get_loop(cls) -> asyncio.AbstractEventLoop:
        """Trả về Playwright loop, khởi tạo nếu chưa có."""
        with cls._lock:
            if cls._thread is None or not cls._thread.is_alive():
                cls._ready.clear()
                cls._thread = threading.Thread(
                    target=cls._thread_main,
                    name="playwright-loop",
                    daemon=True,
                )
                cls._thread.start()
        if not cls._ready.wait(timeout=30):
            raise RuntimeError("Playwright event loop failed to start within 30s.")
        return cls._loop


async def _run_in_pw_loop(coro_func, *args, timeout: float = 300.0):
    """
    Submit coroutine vào Playwright loop, await kết quả từ FastAPI loop.
    Không block FastAPI event loop (dùng run_in_executor).

    timeout=300s (5 phút): đủ cho chapter 80 trang × 900px scroll 600ms/bước
    (worst case: 80×900/250×0.6 = 172s scroll + 60s load + 5s post = ~237s)
    """
    pw_loop = _PlaywrightLoop.get_loop()
    future  = asyncio.run_coroutine_threadsafe(coro_func(*args), pw_loop)
    # run_in_executor wrap blocking future.result() thành awaitable
    main_loop = asyncio.get_event_loop()
    return await main_loop.run_in_executor(None, future.result, timeout)


# ---------------------------------------------------------------------------
# BrowserManager — singleton chạy trong Playwright loop
# (KHÔNG dùng trực tiếp từ FastAPI loop)
# ---------------------------------------------------------------------------

class BrowserManager:
    """
    Singleton Chromium browser.
    Tất cả methods phải được gọi từ trong Playwright loop.
    """
    _playwright = None
    _browser    = None

    @classmethod
    async def get_browser(cls):
        if cls._browser is None or not cls._browser.is_connected():
            await cls._init()
        return cls._browser

    @classmethod
    async def _init(cls) -> None:
        # pyrefly: ignore [missing-import]
        from playwright.async_api import async_playwright
        from app.core.config import settings

        headless = settings.PLAYWRIGHT_HEADLESS
        logger.info(f"[Browser] Launching Chromium (headless={headless})...")
        cls._playwright = await async_playwright().start()
        cls._browser    = await cls._playwright.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        logger.info("[Browser] Chromium ready.")

    @classmethod
    async def _close_impl(cls) -> None:
        if cls._browser:
            try:
                await cls._browser.close()
            except Exception:
                pass
            cls._browser = None
        if cls._playwright:
            try:
                await cls._playwright.stop()
            except Exception:
                pass
            cls._playwright = None
        logger.info("[Browser] Chromium closed.")


# ---------------------------------------------------------------------------
# Slow scroll helper (chạy trong Playwright loop)
# ---------------------------------------------------------------------------

async def _slow_scroll(page) -> None:
    """
    Lướt chậm từ trên xuống để trigger lazy loading.
    250px / 600ms → mô phỏng lướt tay thật.
    """
    scroll_height = await page.evaluate("document.body.scrollHeight")
    current_y = 0
    steps = 0

    while current_y < scroll_height:
        current_y = min(current_y + SCROLL_STEP_PX, scroll_height)
        await page.evaluate(f"window.scrollTo(0, {current_y})")
        await page.wait_for_timeout(SCROLL_DELAY_MS)
        steps += 1
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height > scroll_height:
            scroll_height = new_height
        if steps >= MAX_SCROLL_STEPS:
            logger.warning(f"[Browser] MAX_SCROLL_STEPS={MAX_SCROLL_STEPS} reached at {scroll_height}px — stopping.")
            break

    logger.debug(f"[Browser] Scrolled {steps} steps — height={scroll_height}px")


# ---------------------------------------------------------------------------
# Core scraping logic (chạy trong Playwright loop)
# ---------------------------------------------------------------------------

async def _playwright_scrape_impl(url: str) -> dict:
    """
    Scraping thực sự — PHẢI chạy trong _PlaywrightLoop (ProactorEventLoop).
    Trả về raw dict (chưa có ImageInfo objects).
    """
    browser = await BrowserManager.get_browser()
    page    = await browser.new_page()
    try:
        await page.set_extra_http_headers({
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        })
        await page.set_viewport_size({"width": 390, "height": 844})   # iPhone 15 Pro

        logger.info(f"[Browser] Navigating: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS)

        await _slow_scroll(page)
        await page.wait_for_timeout(POST_SCROLL_WAIT_MS)

        result = await page.evaluate(_EXTRACT_JS)

    finally:
        await page.close()

    raw_title  = result.get("rawTitle", "") or url
    final_url  = result.get("finalUrl", url)
    chapter, chapter_raw = _extract_chapter(raw_title, url)
    domain = urlparse(final_url).netloc.replace("www.", "")

    logger.info(
        f"[Browser] Done: title={raw_title[:60]!r} ch={chapter} "
        f"imgs={len(result.get('images', []))} paras={len(result.get('paragraphs', []))}"
    )

    return {
        "title":       raw_title,
        "chapter":     chapter,
        "chapter_raw": chapter_raw,
        "raw_images":  result.get("images", []),
        "paragraphs":  result.get("paragraphs", []),
        "domain":      domain,
        "final_url":   final_url,
    }


# ---------------------------------------------------------------------------
# Public API (gọi từ FastAPI route)
# ---------------------------------------------------------------------------

async def scrape_with_playwright(url: str) -> dict:
    """
    Public entry point — gọi từ FastAPI event loop.
    Delegate sang Playwright loop (ProactorEventLoop) qua run_in_executor.
    """
    from app.api.routes.scrape import ImageInfo   # lazy import tránh circular

    data = await _run_in_pw_loop(_playwright_scrape_impl, url)

    images = [
        ImageInfo(
            url    = img["url"],
            index  = i,
            width  = img.get("width") or None,
            height = img.get("height") or None,
            alt    = img.get("alt") or None,
        )
        for i, img in enumerate(data.pop("raw_images", []))
    ]
    data["images"] = images
    return data


async def close_browser() -> None:
    """
    Shutdown hook — đóng browser trong Playwright loop.
    Gọi từ FastAPI lifespan (main.py).
    """
    try:
        pw_loop = _PlaywrightLoop.get_loop()
        future  = asyncio.run_coroutine_threadsafe(BrowserManager._close_impl(), pw_loop)
        future.result(timeout=10)
    except Exception as e:
        logger.warning(f"[Browser] close_browser failed: {e}")
