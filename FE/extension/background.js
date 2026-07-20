/**
 * background.js — Manga Translator Extension Service Worker
 *
 * Mục đích: Bắt URL ảnh thật từ network requests trước khi page convert chúng
 * thành blob: URL (kỹ thuật dùng bởi komiic.com và một số site khác).
 *
 * Flow:
 *   1. Page fetch ảnh từ CDN (https://static.komiic.com/...)
 *   2. webRequest.onResponseStarted bắt URL + Content-Type
 *   3. Lưu vào chrome.storage.session theo tabId
 *   4. Content script query GET_CAPTURED_IMAGES → nhận CDN URLs thật
 *   5. Gửi CDN URLs cho backend → download bình thường
 */

// ---------------------------------------------------------------------------
// Capture image URLs từ network
// ---------------------------------------------------------------------------

const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/avif'];

// Bắt tất cả response, lọc ảnh
chrome.webRequest.onResponseStarted.addListener(
  (details) => {
    if (details.tabId <= 0) return;  // ignore service worker/extension requests

    const contentType = details.responseHeaders
      ?.find(h => h.name.toLowerCase() === 'content-type')
      ?.value?.split(';')[0]?.trim() || '';

    const isImage = IMAGE_TYPES.some(t => contentType.startsWith(t));
    if (!isImage) return;

    // Bỏ qua URL quá ngắn hoặc extension tự fetch
    const url = details.url;
    if (url.length < 20 || url.startsWith('chrome-extension://')) return;

    // Lưu vào storage.session (persistent trong session, clear khi browser tắt)
    const key = `tab_${details.tabId}`;
    chrome.storage.session.get(key, (data) => {
      const existing = data[key] || [];
      // Dedup: chỉ thêm nếu chưa có
      if (!existing.includes(url)) {
        const updated = [...existing, url];
        // Giới hạn tối đa 500 URLs/tab để tránh tốn bộ nhớ
        const trimmed = updated.slice(-500);
        chrome.storage.session.set({ [key]: trimmed });
      }
    });
  },
  { urls: ['<all_urls>'] },
  ['responseHeaders']
);

// ---------------------------------------------------------------------------
// Xóa cache khi tab đóng hoặc navigate
// ---------------------------------------------------------------------------

chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.session.remove(`tab_${tabId}`);
});

chrome.webNavigation.onCommitted.addListener((details) => {
  // Navigate sang trang mới → clear cache của tab đó
  if (details.frameId === 0) {
    chrome.storage.session.remove(`tab_${details.tabId}`);
  }
});

// ---------------------------------------------------------------------------
// Message handler — content script giao tiếp với background
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  const tabId = sender.tab?.id;

  if (msg.type === 'GET_CAPTURED_IMAGES') {
    if (!tabId) { sendResponse({ images: [] }); return true; }
    chrome.storage.session.get(`tab_${tabId}`, (data) => {
      const images = data[`tab_${tabId}`] || [];
      sendResponse({ images });
    });
    return true;  // async response
  }

  if (msg.type === 'CLEAR_CAPTURED_IMAGES') {
    if (tabId) chrome.storage.session.remove(`tab_${tabId}`);
    sendResponse({ ok: true });
    return true;
  }
});
