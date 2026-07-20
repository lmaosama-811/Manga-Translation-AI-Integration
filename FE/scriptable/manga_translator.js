// ═══════════════════════════════════════════════════════════════
//  Manga Translator — Scriptable Share Sheet Script
//
//  Cách dùng: Share URL từ Safari → chọn script này trong Scriptable
//
//  SETUP (làm 1 lần):
//    1. Thay GIST_RAW_URL bên dưới
//    2. Import vào Scriptable
//    3. Bật "Show in Share Sheet"
// ═══════════════════════════════════════════════════════════════

// ─── CẤU HÌNH ────────────────────────────────────────────────
const GIST_RAW_URL =
  "https://gist.githubusercontent.com/lmaosama-811/2a580fd74c1312efdc37a739407f092d/raw/backend_url.txt"
const FALLBACK_URL = "http://localhost:8000"
// ─────────────────────────────────────────────────────────────

// ─── KEYCHAIN KEYS (lưu preference) ──────────────────────────
const KC_MODE = "manga_tr_mode"
const KC_MODEL = "manga_tr_model"
const KC_GENRES = "manga_tr_genres"
// ─────────────────────────────────────────────────────────────

// Helpers đọc/ghi Keychain an toàn
function kcGet(key, fallback) {
  try { return Keychain.contains(key) ? Keychain.get(key) : fallback }
  catch (_) { return fallback }
}
function kcSet(key, val) {
  try { Keychain.set(key, val) } catch (_) { }
}

// ─── BƯỚC 0: Lấy URL từ Share Sheet ─────────────────────────
const inputUrl = (args.urls && args.urls.length > 0)
  ? args.urls[0].toString()
  : null

if (!inputUrl) {
  const a = new Alert()
  a.title = "❌ Không có URL"
  a.message = "Hãy share URL từ Safari sang script này."
  a.addAction("OK")
  await a.present()
  Script.complete()
  return
}

// ─── BƯỚC 1: Lấy backend URL từ Gist ─────────────────────────
let backendUrl = FALLBACK_URL
try {
  const r = new Request(GIST_RAW_URL)
  r.timeoutInterval = 5
  const txt = await r.loadString()
  if (txt && txt.trim().startsWith("https://")) backendUrl = txt.trim()
} catch (_) { }

// ─── BƯỚC 2: Chọn MODE ───────────────────────────────────────
const MODES = ["async", "sync", "rich", "novel"]
const MODE_LABELS = {
  async: "⚡ Async  — nhanh, nhiều key song song",
  sync: "🔁 Sync   — tuần tự, ổn định",
  rich: "💎 Rich   — Tier-1 key, chất lượng cao",
  novel: "📖 Novel  — web novel / light novel",
}

const savedMode = kcGet(KC_MODE, "async")
const savedModel = kcGet(KC_MODEL, "gemini-3.1-flash-lite")
const savedGenreStr = kcGet(KC_GENRES, "")  // "action,comedy" hoặc ""

const modeAlert = new Alert()
modeAlert.title = "🎬 Chọn mode dịch"
modeAlert.message = `Lần trước: ${savedMode}`
for (const m of MODES) {
  modeAlert.addAction(MODE_LABELS[m] + (m === savedMode ? "  ✓" : ""))
}
modeAlert.addCancelAction("❌ Huỷ")

const modeIdx = await modeAlert.present()
if (modeIdx === -1) { Script.complete(); return }
const chosenMode = MODES[modeIdx]
kcSet(KC_MODE, chosenMode)

// ─── BƯỚC 3: Chọn MODEL ──────────────────────────────────────
const MODELS = [
  "gemini-3.1-flash-lite",
  "gemini-2.5-flash",
  "gemini-3.5-flash",
]
const MODEL_LABELS = {
  "gemini-3.1-flash-lite": "🌟 Flash Lite   — nhanh, tiết kiệm",
  "gemini-2.5-flash": "⚡ Flash 2.5    — cân bằng",
  "gemini-3.5-flash": "💫 Flash 3.5    — mạnh nhất",
}

const modelAlert = new Alert()
modelAlert.title = "🤖 Chọn model AI"
modelAlert.message = `Lần trước: ${savedModel}`
for (const mdl of MODELS) {
  modelAlert.addAction(MODEL_LABELS[mdl] + (mdl === savedModel ? "  ✓" : ""))
}
modelAlert.addCancelAction("❌ Huỷ")

const modelIdx = await modelAlert.present()
if (modelIdx === -1) { Script.complete(); return }
const chosenModel = MODELS[modelIdx]
kcSet(KC_MODEL, chosenModel)

// ─── BƯỚC 4: Chọn GENRE (đơn giản — 4 SKILLS) ───────────────
// Multi-select bằng cách dùng Alert liên tiếp: toggle từng genre
// Lưu dạng "action,comedy" trong Keychain

const GENRE_LIST = ["action", "comedy", "historical", "mysterious"]
const GENRE_ICONS = { action: "⚔️", comedy: "😄", historical: "🏯", mysterious: "🔮" }

// Parse genres đã lưu
let selectedGenres = new Set(
  savedGenreStr ? savedGenreStr.split(",").filter(g => GENRE_LIST.includes(g)) : []
)

// 1 Alert duy nhất — toggle dạng danh sách
// Mỗi lần nhấn 1 genre thì toggle và hiện lại — đơn giản nhất
let genreDone = false
while (!genreDone) {
  const gAlert = new Alert()
  gAlert.title = "🏷️ Chọn thể loại"
  gAlert.message = selectedGenres.size > 0
    ? `Đã chọn: ${[...selectedGenres].map(g => GENRE_ICONS[g] + g).join(", ")}`
    : "Chưa chọn thể loại nào (sẽ dùng prompt chung)"

  for (const g of GENRE_LIST) {
    const checked = selectedGenres.has(g) ? "✅ " : "⬜ "
    gAlert.addAction(checked + GENRE_ICONS[g] + " " + g.charAt(0).toUpperCase() + g.slice(1))
  }
  gAlert.addAction("✔️  Xác nhận & tiếp tục")
  gAlert.addCancelAction("❌ Huỷ")

  const gIdx = await gAlert.present()

  if (gIdx === -1) { Script.complete(); return }
  if (gIdx < GENRE_LIST.length) {
    // Toggle genre
    const tapped = GENRE_LIST[gIdx]
    if (selectedGenres.has(tapped)) selectedGenres.delete(tapped)
    else selectedGenres.add(tapped)
  } else {
    // "Xác nhận" button
    genreDone = true
  }
}

const chosenGenres = [...selectedGenres]
kcSet(KC_GENRES, chosenGenres.join(","))

// ─── BƯỚC 5: Xác nhận tổng hợp ───────────────────────────────
const shortUrl = inputUrl.length > 55
  ? inputUrl.substring(0, 52) + "..."
  : inputUrl

const summary = [
  `URL:    ${shortUrl}`,
  `Mode:   ${chosenMode}`,
  `Model:  ${chosenModel}`,
  `Genre:  ${chosenGenres.length > 0 ? chosenGenres.join(", ") : "(không chọn)"}`,
  `Server: ${backendUrl}`,
].join("\n")

const confirmAlert = new Alert()
confirmAlert.title = "📤 Xác nhận gửi dịch"
confirmAlert.message = summary
confirmAlert.addAction("✅ Gửi")
confirmAlert.addCancelAction("❌ Huỷ")

if ((await confirmAlert.present()) === -1) { Script.complete(); return }

// ─── BƯỚC 6: Gửi lên /scrape/url ────────────────────────────
const req = new Request(`${backendUrl}/scrape/url`)
req.method = "POST"
req.headers = { "Content-Type": "application/json" }
req.body = JSON.stringify({
  url: inputUrl,
  mode: chosenMode,
  model: chosenModel,
  genre_list: chosenGenres,
})
req.timeoutInterval = 120   // endpoint trả <1s, nhưng buffer cho Cloudflare overhead

let response
let rawBody = ""
try {
  rawBody  = await req.loadString()
  response = JSON.parse(rawBody)
} catch (e) {
  const status = req.response ? req.response.statusCode : "?"
  const a = new Alert()
  a.title = "❌ Lỗi kết nối"
  a.message = (
    `URL: ${backendUrl}/scrape/url\n` +
    `HTTP: ${status}\n` +
    `Lỗi: ${e.message}\n\n` +
    `Response (200 ký tự đầu):\n${rawBody.substring(0, 200)}`
  )
  a.addAction("OK")
  await a.present()
  Script.complete()
  return
}

if (response && response.detail) {
  const a = new Alert()
  a.title = "⚠️ Server báo lỗi"
  // FastAPI 422 trả detail là array [{loc,msg,type}], HTTPException trả string
  a.message = typeof response.detail === "string"
    ? response.detail
    : JSON.stringify(response.detail).substring(0, 300)
  a.addAction("OK")
  await a.present()
  Script.complete()
  return
}

if (!response || response.status !== "accepted") {
  const a = new Alert()
  a.title = "❌ Lỗi không xác định"
  a.message = JSON.stringify(response).substring(0, 300)
  a.addAction("OK")
  await a.present()
  Script.complete()
  return
}


// ─── BƯỚC 7: Mở Reader ───────────────────────────────────────
const localReaderUrl = response.reader_url || ""
const publicReaderUrl = localReaderUrl.replace("http://localhost:8000", backendUrl)

// Primary: tap notification → Safari opens (hoạt động dù extension bị iOS kill)
const n = new Notification()
n.title = "📖 Manga đang xử lý"
n.body  = `👉 Tap để mở trang đọc | Mode: ${chosenMode}`
if (publicReaderUrl) n.openURL = publicReaderUrl   // ← key fix: tap notification = mở Safari
await n.schedule()

// Fallback: Safari.open() nếu iOS chưa kịp kill extension
if (publicReaderUrl) {
  Safari.open(publicReaderUrl)
}

Script.complete()
