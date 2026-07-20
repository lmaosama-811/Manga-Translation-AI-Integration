/**
 * content.js — Manga Translator Extension
 *
 * Flow:
 *   1. Inject floating button (draggable)
 *   2. Hover → "×" close badge xuất hiện ở góc phải
 *   3. Click button → Setup panel (mode/model/genre) trượt ra
 *   4. Confirm → scroll, collect ảnh, POST backend với params
 *   5. Backend trả về reader_url → mở tab mới
 */

// ─── Config ────────────────────────────────────────────────────────────────

const BACKEND_URL = "http://localhost:8000/scrape/submit";

const GENRES = [
  { key: "action",        label: "⚔️ Hành động" },
  { key: "romance",       label: "💕 Lãng mạn"  },
  { key: "comedy",        label: "😄 Hài hước"  },
  { key: "horror",        label: "👻 Kinh dị"   },
  { key: "fantasy",       label: "🔮 Giả tưởng" },
  { key: "adventure",     label: "🗺️ Phiêu lưu" },
  { key: "slice_of_life", label: "🌸 Cuộc sống" },
  { key: "supernatural",  label: "✨ Siêu nhiên"},
  { key: "school",        label: "🏫 Học đường" },
  { key: "sports",        label: "⚽ Thể thao"  },
  { key: "shounen",       label: "🌟 Shounen"   },
  { key: "shoujo",        label: "🌹 Shoujo"    },
  { key: "isekai",        label: "🌀 Isekai"    },
  { key: "mystery",       label: "🔍 Bí ẩn"    },
];

const MODELS = [
  { value: "gemini-3.1-flash-lite", label: "Flash Lite — Nhanh nhất"   },
  { value: "gemini-3.5-flash",      label: "Flash 3.5 — Cân bằng"      },
  { value: "gemini-2.5-flash",      label: "Flash 2.5 — Chất lượng cao" },
];

// Trạng thái người dùng đã chọn trong panel
let _mode   = "async";
let _model  = "gemini-3.1-flash-lite";
let _genres = new Set();

// ─── 1. Inject CSS ─────────────────────────────────────────────────────────

(function injectStyles() {
  if (document.getElementById("mt-styles")) return;
  const style = document.createElement("style");
  style.id = "mt-styles";
  style.textContent = `
    /* ── Floating button ── */
    #mt-float-btn {
      position: fixed;
      bottom: 80px; right: 24px;
      width: 64px; height: 64px;
      border-radius: 50%;
      background: linear-gradient(135deg, #6c63ff, #3ecfcf);
      color: white;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      cursor: pointer;
      z-index: 999999;
      box-shadow: 0 4px 20px rgba(0,0,0,0.35);
      user-select: none;
      font-size: 11px; font-weight: 700; font-family: sans-serif;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    #mt-float-btn:hover {
      transform: scale(1.1);
      box-shadow: 0 6px 28px rgba(108,99,255,0.5);
    }

    /* ── × Close badge ── */
    #mt-close-x {
      position: absolute;
      top: -5px; right: -5px;
      width: 20px; height: 20px;
      background: #ef4444;
      border-radius: 50%;
      font-size: 11px; font-weight: 900;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      opacity: 0;
      transform: scale(0);
      transition: opacity 0.2s ease, transform 0.2s ease;
      z-index: 10;
      box-shadow: 0 2px 8px rgba(239,68,68,0.5);
      line-height: 1;
    }
    #mt-float-btn:hover #mt-close-x {
      opacity: 1;
      transform: scale(1);
    }
    #mt-close-x:hover {
      background: #dc2626 !important;
    }

    /* ── Setup Panel ── */
    #mt-setup-panel {
      position: fixed;
      width: 308px;
      background: rgba(8, 8, 22, 0.97);
      color: #f0f0f5;
      border-radius: 20px;
      padding: 20px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      z-index: 9999998;
      box-shadow: 0 16px 48px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.07);
      backdrop-filter: blur(24px);
      animation: mt-fadeUp 0.22s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    @keyframes mt-fadeUp {
      from { opacity: 0; transform: translateY(14px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0)   scale(1);    }
    }

    .mt-label {
      font-size: 10.5px; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.09em;
      color: #a78bfa; margin-bottom: 8px;
    }
    .mt-divider {
      height: 1px; background: rgba(255,255,255,0.07); margin: 14px 0;
    }

    /* Mode options */
    .mt-mode-opt {
      display: flex; align-items: center; gap: 10px;
      padding: 9px 12px; border-radius: 11px;
      cursor: pointer;
      border: 1.5px solid rgba(255,255,255,0.09);
      transition: all 0.15s; margin-bottom: 6px;
    }
    .mt-mode-opt:hover { background: rgba(255,255,255,0.04); }
    .mt-mode-opt.active {
      background: rgba(108,99,255,0.18);
      border-color: rgba(108,99,255,0.7);
    }
    .mt-radio {
      width: 15px; height: 15px; border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.3);
      flex-shrink: 0; position: relative;
      transition: border-color 0.15s;
    }
    .mt-mode-opt.active .mt-radio { border-color: #6c63ff; }
    .mt-mode-opt.active .mt-radio::after {
      content: ''; position: absolute;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      width: 7px; height: 7px;
      border-radius: 50%; background: #6c63ff;
    }
    .mt-mode-title { font-size: 12.5px; font-weight: 600; }
    .mt-mode-desc  { font-size: 10.5px; opacity: 0.5; margin-top: 1px; }

    /* Model select */
    .mt-model-wrap { position: relative; margin-top: 10px; }
    .mt-select {
      width: 100%; box-sizing: border-box;
      background: rgba(255,255,255,0.06);
      border: 1.5px solid rgba(255,255,255,0.12);
      color: white; border-radius: 10px;
      padding: 8px 32px 8px 12px;
      font-size: 12px; cursor: pointer; outline: none;
      appearance: none; -webkit-appearance: none;
      transition: border-color 0.15s;
    }
    .mt-select:hover { border-color: rgba(108,99,255,0.5); }
    .mt-select option { background: #0e0e1e; color: white; }
    .mt-select-arrow {
      position: absolute; right: 11px; top: 50%;
      transform: translateY(-50%);
      pointer-events: none; opacity: 0.4; font-size: 10px;
    }

    /* Genre chips */
    .mt-chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .mt-chip {
      padding: 4px 10px; border-radius: 20px;
      font-size: 11px; cursor: pointer; user-select: none;
      border: 1.5px solid rgba(255,255,255,0.13);
      background: rgba(255,255,255,0.04);
      color: rgba(255,255,255,0.75);
      transition: all 0.15s; white-space: nowrap;
    }
    .mt-chip:hover { background: rgba(255,255,255,0.1); }
    .mt-chip.active {
      background: rgba(62,207,207,0.18);
      border-color: rgba(62,207,207,0.7);
      color: #5ef0f0;
    }

    /* Buttons */
    .mt-btn-row { display: flex; gap: 8px; margin-top: 16px; }
    .mt-btn-cancel {
      flex: 1; padding: 9px;
      border-radius: 10px;
      border: 1.5px solid rgba(255,255,255,0.15);
      background: transparent; color: rgba(255,255,255,0.6);
      font-size: 12px; font-weight: 600;
      cursor: pointer; transition: all 0.15s;
    }
    .mt-btn-cancel:hover { background: rgba(255,255,255,0.07); color: white; }
    .mt-btn-start {
      flex: 2; padding: 9px;
      border-radius: 10px; border: none;
      background: linear-gradient(135deg, #6c63ff, #3ecfcf);
      color: white; font-size: 12.5px; font-weight: 700;
      cursor: pointer; transition: opacity 0.15s, transform 0.1s;
    }
    .mt-btn-start:hover  { opacity: 0.9; }
    .mt-btn-start:active { transform: scale(0.97); }
    /* Rich mode — gold style */
    .mt-mode-opt[data-mode="rich"] {
      border-color: rgba(251,191,36,0.25);
      background: linear-gradient(135deg, rgba(251,191,36,0.06), rgba(245,158,11,0.06));
    }
    .mt-mode-opt[data-mode="rich"]:hover {
      background: linear-gradient(135deg, rgba(251,191,36,0.1), rgba(245,158,11,0.1));
    }
    .mt-mode-opt[data-mode="rich"].active {
      background: linear-gradient(135deg, rgba(251,191,36,0.2), rgba(245,158,11,0.15));
      border-color: rgba(251,191,36,0.8);
    }
    .mt-mode-opt[data-mode="rich"].active .mt-radio { border-color: #f59e0b; }
    .mt-mode-opt[data-mode="rich"].active .mt-radio::after { background: #f59e0b; }
  `;
  document.head.appendChild(style);
})();

// ─── 2. Inject Floating Button ─────────────────────────────────────────────

(function injectButton() {
  if (document.getElementById("mt-float-btn")) return;

  const btn = document.createElement("div");
  btn.id = "mt-float-btn";
  btn.innerHTML = `
    <div id="mt-close-x" title="Tắt extension">✕</div>
    <span class="mt-icon" style="font-size:22px;">🔍</span>
    <div id="mt-label">Dịch</div>
  `;

  // ×  dismiss the entire extension for this session
  btn.querySelector("#mt-close-x").addEventListener("click", (e) => {
    e.stopPropagation();
    closeSetupPanel();
    btn.remove();
    try { sessionStorage.setItem("mt-dismissed", "1"); } catch (_) {}
  });

  makeDraggable(btn);

  btn.addEventListener("click", (e) => {
    if (e.target.id === "mt-close-x") return;
    if (btn._wasDragged) return;
    toggleSetupPanel();
  });

  document.body.appendChild(btn);
})();

// ─── 3. Drag logic ─────────────────────────────────────────────────────────

function makeDraggable(el) {
  let sx, sy, il, it, dragging = false, moved = false;

  el.addEventListener("mousedown", (e) => {
    if (e.target.id === "mt-close-x") return;
    dragging = true; moved = false;
    sx = e.clientX; sy = e.clientY;
    const r = el.getBoundingClientRect();
    il = r.left; it = r.top;
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - sx;
    const dy = e.clientY - sy;
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) moved = true;
    el.style.left   = `${il + dx}px`;
    el.style.top    = `${it + dy}px`;
    el.style.right  = "auto";
    el.style.bottom = "auto";
  });

  document.addEventListener("mouseup", () => {
    el._wasDragged = moved;
    dragging = false;
    setTimeout(() => { el._wasDragged = false; }, 120);
  });
}

// ─── 4. Setup Panel ────────────────────────────────────────────────────────

function closeSetupPanel() {
  const p = document.getElementById("mt-setup-panel");
  if (p) p.remove();
}

function toggleSetupPanel() {
  if (document.getElementById("mt-setup-panel")) { closeSetupPanel(); return; }
  showSetupPanel();
}

function showSetupPanel() {
  const panel = document.createElement("div");
  panel.id = "mt-setup-panel";

  panel.innerHTML = `
    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
      <div style="font-size:15px;font-weight:700;">⚙️ Cài đặt dịch</div>
      <div id="mt-panel-x"
           style="cursor:pointer;opacity:0.55;font-size:15px;padding:3px 8px;
                  border-radius:8px;transition:all 0.15s;"
           onmouseover="this.style.opacity='1';this.style.background='rgba(255,255,255,0.08)'"
           onmouseout="this.style.opacity='0.55';this.style.background='transparent'">✕</div>
    </div>

    <!-- Chế độ dịch -->
    <div class="mt-label">Chế độ dịch</div>

    <div class="mt-mode-opt ${_mode === 'async' ? 'active' : ''}" data-mode="async">
      <div class="mt-radio"></div>
      <div>
        <div class="mt-mode-title">⚡ Async — Song song</div>
        <div class="mt-mode-desc">Nhanh · Flash Lite · Tự retry khi RPM</div>
      </div>
    </div>

    <div class="mt-mode-opt ${_mode === 'sync' ? 'active' : ''}" data-mode="sync">
      <div class="mt-radio"></div>
      <div>
        <div class="mt-mode-title">🔄 Sync — Tuần tự</div>
        <div class="mt-mode-desc">Ổn định hơn · Chọn được model</div>
      </div>
    </div>

    <div class="mt-mode-opt ${_mode === 'rich' ? 'active' : ''}" data-mode="rich">
      <div class="mt-radio"></div>
      <div>
        <div class="mt-mode-title">💎 Người giàu — Tier 1</div>
        <div class="mt-mode-desc">20 workers · Key Tier 1 · Tốc độ tối đa · Chọn model</div>
      </div>
    </div>

    <div class="mt-mode-opt ${_mode === 'novel' ? 'active' : ''}" data-mode="novel">
      <div class="mt-radio"></div>
      <div>
        <div class="mt-mode-title">📖 Novel — Dịch text</div>
        <div class="mt-mode-desc">Web novel · Cả chương 1 lần · Hiển thị văn bản</div>
      </div>
    </div>

    <!-- Model (hiện khi sync, rich hoặc novel) -->
    <div id="mt-model-section" style="display:${(_mode === 'sync' || _mode === 'rich' || _mode === 'novel') ? 'block' : 'none'};margin-top:10px;">
      <div class="mt-label" style="margin-bottom:6px;">Model</div>
      <div class="mt-model-wrap">
        <select class="mt-select" id="mt-model-select">
          ${MODELS.map(m => `<option value="${m.value}" ${_model === m.value ? 'selected' : ''}>${m.label}</option>`).join("")}
        </select>
        <div class="mt-select-arrow">▾</div>
      </div>
    </div>

    <div class="mt-divider"></div>

    <!-- Thể loại -->
    <div class="mt-label">
      Thể loại
      <span style="font-weight:400;opacity:0.4;font-size:10px;text-transform:none;letter-spacing:0;"> — tuỳ chọn</span>
    </div>
    <div class="mt-chips" id="mt-chips">
      ${GENRES.map(g => `
        <div class="mt-chip ${_genres.has(g.key) ? 'active' : ''}" data-genre="${g.key}">${g.label}</div>
      `).join("")}
    </div>

    <!-- Buttons -->
    <div class="mt-btn-row">
      <button class="mt-btn-cancel" id="mt-cancel-btn">Hủy</button>
      <button class="mt-btn-start"  id="mt-start-btn">🚀 Bắt đầu dịch</button>
    </div>
  `;

  // Vị trí panel gần button
  const btn     = document.getElementById("mt-float-btn");
  const btnRect = btn ? btn.getBoundingClientRect() : { top: 200, left: window.innerWidth - 340 };
  const panelH  = 440;
  const top     = Math.max(10, btnRect.top - panelH - 8);
  const left    = Math.max(10, btnRect.left - 316);
  panel.style.top  = `${top}px`;
  panel.style.left = `${left}px`;

  document.body.appendChild(panel);

  // Event: đóng
  panel.querySelector("#mt-panel-x").addEventListener("click", closeSetupPanel);
  panel.querySelector("#mt-cancel-btn").addEventListener("click", closeSetupPanel);

  // Event: chọn mode
  panel.querySelectorAll(".mt-mode-opt").forEach(opt => {
    opt.addEventListener("click", () => {
      panel.querySelectorAll(".mt-mode-opt").forEach(o => o.classList.remove("active"));
      opt.classList.add("active");
      _mode = opt.dataset.mode;
      // Hiện model selector khi chọn sync, rich hoặc novel
      const modelSec = panel.querySelector("#mt-model-section");
      modelSec.style.display = (_mode === "sync" || _mode === "rich" || _mode === "novel") ? "block" : "none";
    });
  });

  // Event: chọn model
  panel.querySelector("#mt-model-select").addEventListener("change", (e) => {
    _model = e.target.value;
  });

  // Event: chọn genre chips
  panel.querySelectorAll(".mt-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const key = chip.dataset.genre;
      if (_genres.has(key)) { _genres.delete(key); chip.classList.remove("active"); }
      else                  { _genres.add(key);    chip.classList.add("active");    }
    });
  });

  // Event: bắt đầu dịch
  panel.querySelector("#mt-start-btn").addEventListener("click", () => {
    closeSetupPanel();
    handleTranslateClick();
  });
}

// ─── 5. Main translate handler ─────────────────────────────────────────────

async function handleTranslateClick() {
  setButtonState("loading");
  showToast("⏳ Đang quét trang...");

  try {
    const metadata = extractMetadata();
    showToast(`📖 ${metadata.title} — Chap ${metadata.chapter}`);

    // ── Novel mode: thu thập text, bỏ qua ảnh ──────────────────────────────
    if (_mode === "novel") {
      showToast("📖 Đang thu thập nội dung novel...");
      const paragraphs = collectNovelText();

      if (paragraphs.length === 0) {
        showToast("❌ Không tìm thấy nội dung novel! Thử load hết trang rồi bấm Dịch.", "error");
        setButtonState("idle");
        return;
      }

      showToast(`📝 ${paragraphs.length} đoạn văn — Đang gửi backend...`);
      const result = await sendToBackend(metadata, [], paragraphs);

      if (result.reader_url) {
        showToast("✅ Đang mở trang đọc...");
        window.open(result.reader_url, "_blank");
      }
      setButtonState("done");
      return;
    }

    // ── Manga mode: scroll + thu thập ảnh ──────────────────────────────────
    if (isPaginatedReader()) {
      showToast("📖 Phát hiện chế độ phân trang — đang tự chuyển trang...");
      const imageUrls = await autoPaginateCollect();

      if (imageUrls.length === 0) {
        showToast("❌ Không thu thập được ảnh nào! Kiểm tra lại trang.", "error");
        setButtonState("idle");
        return;
      }

      showToast(`📡 ${imageUrls.length} ảnh — Đang gửi backend...`);
      const result = await sendToBackend(metadata, imageUrls, []);

      if (result.reader_url) {
        showToast("✅ Đang mở trang đọc...");
        window.open(result.reader_url, "_blank");
      }
      showResultPanel(metadata, imageUrls, result);
      setButtonState("done");
      return;
    }

    // ── Scroll-based site: scroll + thu thập ảnh ───────────────────────────
    showToast("📜 Đang cuộn trang để load ảnh...");
    await scrollToLoadAll();

    // Ưu tiên dùng CDN URLs đã capture từ network (background service worker).
    let imageUrls = await queryCapturedImages();

    if (imageUrls.length > 0) {
      console.log(`[MT] Dùng ${imageUrls.length} CDN URL từ network capture (background script)`);
      showToast(`🌐 ${imageUrls.length} ảnh từ CDN · Mode: ${_mode.toUpperCase()}`);
    } else {
      // Fallback: scan DOM (hầu hết các site thông thường)
      imageUrls = collectImageUrls();
      console.log(`[MT] Dùng ${imageUrls.length} ảnh từ DOM scan`);
      showToast(`🖼️ Tìm thấy ${imageUrls.length} ảnh · Mode: ${_mode.toUpperCase()}`);
    }

    if (imageUrls.length === 0) {
      showToast("❌ Không tìm thấy ảnh nào! Thử scroll hết trang rồi bấm Dịch.", "error");
      setButtonState("idle");
      return;
    }

    showToast("📡 Đang gửi về backend...");
    const result = await sendToBackend(metadata, imageUrls, []);

    if (result.reader_url) {
      showToast("✅ Đang mở trang đọc...");
      window.open(result.reader_url, "_blank");
    }

    showResultPanel(metadata, imageUrls, result);
    setButtonState("done");

  } catch (err) {
    console.error("[MT Extension]", err);
    showToast(`❌ Lỗi: ${err.message}`, "error");
    setButtonState("idle");
  }
}

// ─── 6. Extract metadata từ DOM ───────────────────────────────────────────

function extractMetadata() {
  const titleSelectors = [
    ".breadcrumb li:nth-child(2) a",
    "h1.title-detail",
    ".detail-info h1",
    "h1[data-v-46da5678]",
    "h1",
    "title",
  ];
  const chapterSelectors = [
    ".breadcrumb li:last-child",
    "h2.title-chapter",
    ".chapter-title",
    ".chakra-breadcrumb__list li:last-child",
    "[class*='chapter']",
  ];

  let title = "Unknown";
  for (const sel of titleSelectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim()) { title = el.textContent.trim(); break; }
  }
  if (title === "Unknown") {
    title = document.title.split("|")[0].split("-")[0].trim();
  }

  let chapterRaw = "0";
  for (const sel of chapterSelectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent.match(/\d+/)) { chapterRaw = el.textContent; break; }
  }

  const chapterMatch = chapterRaw.match(/(\d+)/);
  const chapter = chapterMatch ? parseInt(chapterMatch[1]) : 0;

  return {
    title:        title.replace(/\s+/g, " ").trim(),
    chapter:      chapter,
    chapter_raw:  chapterRaw.trim(),
    url:          window.location.href,
    domain:       window.location.hostname,
    extracted_at: new Date().toISOString(),
  };
}

// ─── 7. Scroll lazy-load ──────────────────────────────────────────────────

async function scrollToLoadAll() {
  const delay    = ms => new Promise(r => setTimeout(r, ms));
  const total    = document.body.scrollHeight;
  const step     = window.innerHeight;

  for (let y = 0; y < total; y += step) {
    window.scrollTo(0, y);
    await delay(300);
  }
  window.scrollTo(0, 0);
  await delay(500);
}

// ─── 7b. Pagination detection & auto-click ────────────────────────────────

// Selectors cho nút "trang sau" — thử theo thứ tự ưu tiên.
// Dùng :not([disabled]) để bỏ qua nút đã disabled ở trang cuối.
const NEXT_BTN_SELECTORS = [
  'a[rel="next"]',
  '[class*="next-page"]:not([disabled])',
  '[class*="page-next"]:not([disabled])',
  '[class*="btn-next"]:not([disabled])',
  '[class*="next-btn"]:not([disabled])',
  '.next:not([disabled])',
  '.nextBtn:not([disabled])',
  '[aria-label*="next" i]:not([disabled])',
  '[title*="next" i]:not([disabled])',
  '[title*="sau" i]:not([disabled])',
  '[aria-label*="sau" i]:not([disabled])',
  'button[class*="right"]:not([disabled])',
  'button[class*="forward"]:not([disabled])',
  'a[class*="right"]:not([disabled])',
];

function findNextButton() {
  // Pass 1: CSS selector list
  for (const sel of NEXT_BTN_SELECTORS) {
    try {
      const els = document.querySelectorAll(sel);
      for (const el of els) {
        if (el.offsetParent !== null && !el.disabled) return el;
      }
    } catch (_) {}
  }

  // Pass 2: scan text/innerHTML của tất cả button/a/role=button
  const NEXT_TEXTS  = ['>', '>>', '→', '›', '»', 'next', 'next page', 'sau', 'trang sau'];
  const NEXT_ICONS  = ['chevron-right', 'arrow-right', 'angle-right', 'caret-right',
                       'icon-next', 'icon-right', 'fa-next'];
  const candidates  = document.querySelectorAll('button, a[href], [role="button"]');

  for (const el of candidates) {
    if (!el.offsetParent || el.disabled) continue;
    const text  = (el.textContent || '').trim().toLowerCase();
    const label = (el.getAttribute('aria-label') || '').toLowerCase();
    const title = (el.getAttribute('title') || '').toLowerCase();
    const html  = el.innerHTML.toLowerCase();

    const matchText = NEXT_TEXTS.some(t => text === t || label === t || title === t);
    const matchIcon = NEXT_ICONS.some(ic => html.includes(ic));

    if (matchText || matchIcon) return el;
  }

  return null;
}


function isPaginatedReader() {
  const nextBtn = findNextButton();
  if (!nextBtn) return false;

  // Đếm số ảnh lớn hiện có trong DOM.
  // Scroll-based site thường có >= 10 ảnh; pagination chỉ có 1-3.
  const largeImgCount = Array.from(document.querySelectorAll('img')).filter(img => {
    const w = img.naturalWidth  || img.width  || 0;
    const h = img.naturalHeight || img.height || 0;
    return w > 200 && h > 200;
  }).length;

  console.log(`[MT] isPaginatedReader: nextBtn=${!!nextBtn}, largeImgs=${largeImgCount}`);
  return largeImgCount <= 5;
}


async function waitForPageChange(prevUrl, prevMainSrc, timeout = 4000) {
  const delay = ms => new Promise(r => setTimeout(r, ms));

  return new Promise(resolve => {
    let settled = false;
    const done = (reason) => {
      if (settled) return;
      settled = true;
      obs.disconnect();
      clearInterval(urlPoller);
      clearTimeout(timer);
      resolve(reason);
    };

    // 1. Poll URL thay doi (SPA navigation)
    const urlPoller = setInterval(() => {
      if (window.location.href !== prevUrl) done('url_changed');
    }, 100);

    // 2. MutationObserver: watch src thay doi tren img lon, hoac them img moi vao DOM
    const obs = new MutationObserver((mutations) => {
      for (const mut of mutations) {
        if (mut.type === 'attributes' && mut.attributeName === 'src') {
          const el = mut.target;
          if (el.tagName !== 'IMG') continue;
          const newSrc = el.src || '';
          if (newSrc && newSrc !== prevMainSrc && !newSrc.startsWith('data:')) {
            done('img_src_changed');
          }
        }
        if (mut.type === 'childList') {
          for (const node of mut.addedNodes) {
            if (!node.querySelectorAll) continue;
            const hasImg = node.tagName === 'IMG' || node.querySelectorAll('img').length > 0;
            if (hasImg) { done('new_img_node'); return; }
          }
        }
      }
    });
    obs.observe(document.body, {
      childList: true, subtree: true,
      attributes: true, attributeFilter: ['src'],
    });

    const timer = setTimeout(() => done('timeout'), timeout);

    // Kiem tra ngay (co the da doi truoc khi attach)
    delay(150).then(() => {
      if (window.location.href !== prevUrl) done('url_imm');
      const cur = getMainImageSrc();
      if (cur && cur !== prevMainSrc) done('src_imm');
    });
  });
}


function getMainImageSrc() {
  let best = null, bestArea = 0;
  document.querySelectorAll('img').forEach(img => {
    if (!img.offsetParent) return;
    const w = img.naturalWidth  || img.width  || 0;
    const h = img.naturalHeight || img.height || 0;
    if (w * h > bestArea) { bestArea = w * h; best = img; }
  });
  if (!best) return '';
  return best.src || best.getAttribute('data-src') || '';
}


async function autoPaginateCollect() {
  const MAX_PAGES = 120;
  const delay     = ms => new Promise(r => setTimeout(r, ms));

  const collected    = new Map(); // url -> ImageInfo, dedup
  let   pageNum      = 1;
  let   noGainStreak = 0;

  const snapshot = () => {
    let added = 0;
    for (const img of collectImageUrls()) {
      if (!collected.has(img.url)) { collected.set(img.url, img); added++; }
    }
    return added;
  };

  // Thu thap trang dau
  snapshot();
  showToast(`\ud83d\udcc4 Trang 1 \u2014 ${collected.size} \u1ea3nh`);
  console.log('[MT Paginate] Page 1:', collected.size, 'images');

  while (pageNum < MAX_PAGES) {
    const nextBtn = findNextButton();
    if (!nextBtn) {
      console.log('[MT Paginate] Khong tim thay nut Next \u2014 ket thuc.');
      break;
    }
    if (nextBtn.disabled || nextBtn.getAttribute('disabled') !== null ||
        nextBtn.classList.contains('disabled')) {
      console.log('[MT Paginate] Nut Next bi disabled \u2014 het chapter.');
      break;
    }

    const prevUrl     = window.location.href;
    const prevImgSrc  = getMainImageSrc();

    nextBtn.click();
    pageNum++;

    // Doi trang moi load (URL change hoac img src change)
    const reason = await waitForPageChange(prevUrl, prevImgSrc);
    await delay(300); // buffer nho cho anh fully render

    const gained = snapshot();
    const newUrl = window.location.href;

    console.log(
      `[MT Paginate] Page ${pageNum}: +${gained} anh (tong ${collected.size}) | reason=${reason} | URL_changed=${newUrl !== prevUrl}`
    );
    showToast(`\ud83d\udcc4 Trang ${pageNum} \u2014 t\u1ed5ng ${collected.size} \u1ea3nh`);

    if (gained === 0 && newUrl === prevUrl) {
      noGainStreak++;
      if (noGainStreak >= 3) {
        console.log('[MT Paginate] 3 lan lien tiep khong co anh moi \u2014 dung.');
        break;
      }
    } else {
      noGainStreak = 0;
    }
  }

  console.log(`[MT Paginate] DOM scan: ${pageNum} trang, ${collected.size} anh.`);

  // Cross-check voi CDN captures tu background script.
  // Neu user da browse thu cong truoc do, background script co the da co du URL.
  const cdnUrls = await queryCapturedImages();
  console.log(`[MT Paginate] CDN captures: ${cdnUrls.length} anh.`);

  if (cdnUrls.length > collected.size) {
    showToast(`\ud83c\udf10 CDN capture: ${cdnUrls.length} \u1ea3nh (hon DOM scan ${collected.size})`);
    return cdnUrls.map((img, i) => ({ ...img, index: i }));
  }

  return Array.from(collected.values()).map((img, i) => ({ ...img, index: i }));
}

// ─── 8. Collect image URLs ────────────────────────────────────────────────

function collectImageUrls() {
  const seen = new Set();
  const urls = [];

  /**
   * Chuẩn hoá URL về dạng tuyệt đối (absolute URL).
   * Xử lý các trường hợp:
   *   "//cdn.komiic.com/..." → "https://cdn.komiic.com/..."
   *   "/api/image/..."       → "https://komiic.com/api/image/..."
   *   "https://..."          → giữ nguyên
   */
  function normalizeUrl(raw) {
    try {
      return new URL(raw, window.location.href).href;
    } catch {
      return raw;  // fallback: giữ nguyên nếu parse thất bại
    }
  }

  document.querySelectorAll("img").forEach(img => {
    const rawCandidates = [
      img.getAttribute("data-src"),
      img.getAttribute("data-lazy-src"),
      img.getAttribute("data-original"),
      img.getAttribute("data-url"),
      img.src,
    ].filter(Boolean);

    for (const rawUrl of rawCandidates) {
      if (!rawUrl || rawUrl.startsWith("data:") || rawUrl.length < 10) continue;

      // Resolve về absolute URL trước khi check/dedup
      const url = normalizeUrl(rawUrl);
      if (seen.has(url)) continue;

      const u = url.toLowerCase();
      const isMangaLike =
        u.includes(".jpg") || u.includes(".jpeg") ||
        u.includes(".png") || u.includes(".webp") ||
        u.includes("image") || u.includes("manga") ||
        u.includes("chapter") || u.includes("page");

      const w = img.naturalWidth  || img.width  || 999;
      const h = img.naturalHeight || img.height || 999;

      if (isMangaLike || (w > 100 && h > 100)) {
        seen.add(url);
        urls.push({ url, index: urls.length, width: img.naturalWidth || null, height: img.naturalHeight || null, alt: img.alt || null });
      }
    }
  });

  return urls;
}

// ─── 8. Query background script for CDN URLs ──────────────────────────────

/**
 * Hỏi background service worker về CDN URLs đã capture từ network requests.
 * Background script dùng chrome.webRequest để bắt URL thật TRƯỚC KHI page
 * convert chúng sang blob: URL (kỹ thuật anti-scraping của komiic.com, ...).
 *
 * Trả về list ImageInfo giống collectImageUrls() để compatible với phần còn lại.
 * Nếu background script không có sẵn → trả về [] (sẽ fallback về DOM scan).
 */
async function queryCapturedImages() {
  try {
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => resolve([]), 1500);  // timeout 1.5s

      chrome.runtime.sendMessage({ type: 'GET_CAPTURED_IMAGES' }, (response) => {
        clearTimeout(timer);
        if (chrome.runtime.lastError) {
          console.warn('[MT] Background script không khả dụng:', chrome.runtime.lastError.message);
          resolve([]);
          return;
        }
        const urls = (response?.images || [])
          .filter(url => url && (url.startsWith('http://') || url.startsWith('https://')))
          // Lọc ảnh có vẻ là manga page (bỏ qua icon, avatar, ...)
          .filter(url => {
            const u = url.toLowerCase();
            // Bỏ qua các pattern rõ ràng không phải trang truyện
            const skipPatterns = ['favicon', 'logo', 'avatar', 'icon', 'thumbnail', 'banner'];
            if (skipPatterns.some(p => u.includes(p))) return false;
            return true;
          })
          .map((url, index) => ({ url, index, width: null, height: null, alt: null }));

        console.log(`[MT] queryCapturedImages: ${urls.length} CDN URL từ background`);
        resolve(urls);
      });
    });
  } catch (e) {
    console.warn('[MT] queryCapturedImages error:', e);
    return [];
  }
}

// ─── 9. Collect image URLs từ DOM ─────────────────────────────────────────


/**
 * Chuyển blob: URL → data: URL bằng cách fetch blob từ browser memory.
 * blob: URL chỉ tồn tại trong browser tab — backend không download được.
 *
 * komiic.com và một số site dùng URL.createObjectURL(blob) để render ảnh
 * mà không expose URL thật. Content script có quyền fetch blob: URL của trang.
 */
async function blobUrlToDataUrl(blobUrl) {
  try {
    const response = await fetch(blobUrl);
    const blob     = await response.blob();
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);  // "data:image/webp;base64,..."
      reader.onerror   = reject;
      reader.readAsDataURL(blob);
    });
  } catch (e) {
    console.warn('[MT] Không thể convert blob URL:', blobUrl, e);
    return blobUrl;  // fallback: giữ nguyên (sẽ bị skip ở backend)
  }
}

/**
 * Resolve tất cả blob: URLs trong danh sách ảnh → inline base64 data: URLs.
 * Ảnh với URL thường (http/https) được giữ nguyên.
 */
async function resolveImageUrls(imageUrls) {
  const blobCount = imageUrls.filter(img => img.url.startsWith('blob:')).length;
  if (blobCount > 0) console.log(`[MT] Đang convert ${blobCount} blob: URL→base64...`);

  return await Promise.all(imageUrls.map(async (img) => {
    if (img.url.startsWith('blob:')) {
      const dataUrl = await blobUrlToDataUrl(img.url);
      return { ...img, url: dataUrl };
    }
    return img;
  }));
}

// ─── 9b. Collect novel text ───────────────────────────────────────────────

/**
 * collectNovelText() — Thu thập các đoạn văn từ trang novel.
 *
 * Chiến lược:
 *   1. Thử các CSS selector phổ biến của web đọc novel (xếp theo ưu tiên)
 *   2. Tìm container có nội dung dài nhất (fallback)
 *   3. Lấy tất cả <p> hoặc dòng text trong container
 *   4. Lọc dòng quá ngắn (< 10 ký tự) — thường là nút bấm hay metadata
 */
function collectNovelText() {
  const CONTAINER_SELECTORS = [
    "#chapter-content",       // metruyenchu, tangthuvien
    ".chapter-content",
    ".content-chapter",
    ".reading-content",
    "#article-content",
    ".article-content",
    ".novel-content",
    ".chapter-text",
    ".read-content",
    ".box-chap",              // truyen.tangthuvien.vn
    "#content",
    ".content",
    "article",
    "[class*='chapter']",
  ];

  let container = null;
  for (const sel of CONTAINER_SELECTORS) {
    const el = document.querySelector(sel);
    if (el && el.innerText.trim().length > 200) { container = el; break; }
  }

  // Fallback: block lớn nhất trên trang
  if (!container) {
    let best = null, bestLen = 0;
    document.querySelectorAll("div, section, article").forEach(el => {
      const len = el.innerText.trim().length;
      if (len > bestLen && len > 500) { bestLen = len; best = el; }
    });
    container = best || document.body;
  }

  const paragraphs = [];
  const pEls = container.querySelectorAll("p");

  if (pEls.length > 0) {
    pEls.forEach(p => {
      const text = p.innerText.trim();
      if (text.length >= 10) paragraphs.push(text);
    });
  } else {
    // Fallback: split by newline nếu không có <p> tags
    container.innerText.trim().split("\n").forEach(line => {
      const text = line.trim();
      if (text.length >= 10) paragraphs.push(text);
    });
  }

  const tag = container.tagName + (container.id ? "#" + container.id : "");
  console.log(`[MT] Novel: ${paragraphs.length} đoạn trong <${tag}>`);
  return paragraphs;
}

// ─── 10. Backend POST ─────────────────────────────────────────────────────

async function sendToBackend(metadata, imageUrls, textParagraphs = []) {
  // Resolve blob: URLs → inline base64 data: URLs trước khi gửi backend (manga only)
  const resolvedUrls = imageUrls.length > 0 ? await resolveImageUrls(imageUrls) : [];
  if (resolvedUrls.length > 0)
    console.log(`[MT] Resolved ${resolvedUrls.length} URLs (đã convert blob: nếu có)`);

  const payload = {
    metadata,
    image_urls:      resolvedUrls,
    text_paragraphs: textParagraphs,          // novel: đầy đoạn văn; manga: []
    total_pages:     resolvedUrls.length || textParagraphs.length,
    cookies:         document.cookie,
    user_agent:      navigator.userAgent,
    // Cài đặt người dùng chọn trong Setup Panel
    mode:            _mode,
    model:           (_mode === "sync" || _mode === "rich" || _mode === "novel") ? _model : undefined,
    genre_list:      Array.from(_genres),
  };

  console.log("[MT Extension] Submitting:", {
    mode: _mode, model: payload.model,
    genres: payload.genre_list,
    images: resolvedUrls.length,
    paragraphs: textParagraphs.length,
  });

  const res = await fetch(BACKEND_URL, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Backend lỗi ${res.status}: ${text}`);
  }

  return await res.json();
}

// ─── 10. UI Helpers ───────────────────────────────────────────────────────

function setButtonState(state) {
  const btn   = document.getElementById("mt-float-btn");
  if (!btn) return;
  const label = document.getElementById("mt-label");
  const icon  = btn.querySelector(".mt-icon");

  const states = {
    idle:    { i: "🔍", t: "Dịch",  bg: "linear-gradient(135deg, #6c63ff, #3ecfcf)"  },
    loading: { i: "⏳", t: "...",    bg: "linear-gradient(135deg, #f59e0b, #ef4444)"  },
    done:    { i: "✅", t: "Xong!", bg: "linear-gradient(135deg, #22c55e, #16a34a)" },
  };

  const s = states[state] || states.idle;
  if (icon)  icon.textContent  = s.i;
  if (label) label.textContent = s.t;
  btn.style.background = s.bg;
}

function showToast(msg, type = "info") {
  const existing = document.getElementById("mt-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.id = "mt-toast";
  toast.textContent = msg;

  Object.assign(toast.style, {
    position:       "fixed",
    bottom:         "160px",
    right:          "16px",
    maxWidth:       "280px",
    background:     type === "error" ? "rgba(239,68,68,0.95)" : "rgba(20,20,40,0.95)",
    color:          "white",
    padding:        "10px 16px",
    borderRadius:   "12px",
    fontSize:       "13px",
    fontFamily:     "sans-serif",
    zIndex:         "9999999",
    boxShadow:      "0 4px 16px rgba(0,0,0,0.35)",
    backdropFilter: "blur(10px)",
    border:         "1px solid rgba(255,255,255,0.08)",
    transition:     "opacity 0.3s",
    lineHeight:     "1.5",
  });

  document.body.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 4000);
}

function showResultPanel(metadata, imageUrls, backendResult) {
  const existing = document.getElementById("mt-result-panel");
  if (existing) existing.remove();

  const panel = document.createElement("div");
  panel.id = "mt-result-panel";

  const readerUrl = backendResult.reader_url || "";
  const jobId     = backendResult.job_id     || "";
  const modeLabel = _mode === "async" ? "⚡ Async" : "🔄 Sync";

  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <strong style="font-size:14px;">🚀 Đang dịch nền</strong>
      <span id="mt-result-close" style="cursor:pointer;font-size:18px;opacity:0.6;padding:2px 6px;">✕</span>
    </div>
    <div style="font-size:12px;line-height:1.9;color:#ccc;">
      <div>📖 <b>Truyện:</b> ${metadata.title}</div>
      <div>📄 <b>Chapter:</b> ${metadata.chapter}</div>
      <div>🖼️ <b>Tổng trang:</b> ${backendResult.total_pages || imageUrls.length}</div>
      <div>⚙️ <b>Chế độ:</b> ${modeLabel}${_mode === 'sync' ? ' · ' + _model.replace('gemini-', '') : ''}</div>
      ${_genres.size ? `<div>🏷️ <b>Thể loại:</b> ${Array.from(_genres).join(', ')}</div>` : ""}
      <div>🔑 <b>Job:</b> <span style="font-family:monospace;font-size:10px;opacity:0.6;">${jobId.substring(0,12)}...</span></div>
      ${readerUrl ? `
      <div style="margin-top:10px;">
        <a href="${readerUrl}" target="_blank"
           style="display:block;padding:8px 12px;
                  background:linear-gradient(135deg,#6c63ff,#3ecfcf);
                  color:white;border-radius:8px;text-decoration:none;
                  font-weight:600;text-align:center;font-size:12px;">
          📖 Mở trang đọc
        </a>
      </div>` : ""}
    </div>
  `;

  Object.assign(panel.style, {
    position:       "fixed",
    bottom:         "160px",
    right:          "16px",
    width:          "280px",
    background:     "rgba(8,8,22,0.97)",
    color:          "white",
    padding:        "16px",
    borderRadius:   "16px",
    fontSize:       "13px",
    fontFamily:     "sans-serif",
    zIndex:         "9999998",
    boxShadow:      "0 8px 32px rgba(0,0,0,0.55)",
    backdropFilter: "blur(20px)",
    border:         "1px solid rgba(255,255,255,0.08)",
  });

  panel.querySelector("#mt-result-close").addEventListener("click", () => panel.remove());
  document.body.appendChild(panel);
}
