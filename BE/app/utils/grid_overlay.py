"""
Module: app.utils.grid_overlay
Description: Draws a semi-transparent coordinate reference grid onto a PIL Image
             before it is encoded and sent to a Vision LLM (e.g. Gemini).

Why this works
--------------
VLMs are fundamentally bad at estimating pixel distances from raw images because
they process images through patch embeddings, not a pixel ruler.  However, they
are extremely good at reading text labels.  By painting a lightweight, numbered
grid directly on the image we give the model a visual ruler it can simply read:

  "The speech bubble's top edge sits between the Y=200 and Y=300 grid lines,
   closer to 250.  Its left edge is on the X=600 line."

This converts coordinate estimation from a spatial-reasoning task (hard) into a
text-reading task (easy), improving bbox accuracy significantly.

Grid design
-----------
- Spacing: every GRID_STEP units on the [0, 1000] normalized scale.
- Lines:   semi-transparent light-gray overlay (does not obscure manga content).
- Labels:  small black-on-white numbers at regular intervals along all four edges
           so the model can triangulate from any corner.
- Output:  the annotated image is returned both as a PIL Image and as a base64
           JPEG string ready for the VLM API.

Usage
-----
    from app.utils.grid_overlay import apply_grid_overlay

    annotated_b64 = apply_grid_overlay(orig_pil_image)
    # pass annotated_b64 to vlm_model.translate_with_retry(...)
"""

import io
import base64
import logging
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────

# How many tick lines to draw (spaced evenly across the 0-1000 range)
GRID_DIVISIONS = 50          # → lines at 0, 100, 200, … 900, 1000

# Visual style
GRID_LINE_COLOR = (160, 160, 160)  # light gray lines
GRID_LINE_ALPHA = 60               # 0-255, higher = more visible
LABEL_FONT_SIZE_DIVISOR = 60       # label font size = max(10, image_short_side // divisor)
LABEL_PADDING = 4                  # pixels between label text and image edge
LABEL_BG_COLOR = (255, 255, 255)   # white label background for readability
LABEL_TEXT_COLOR = (20, 20, 20)    # near-black text

# JPEG quality for the base64-encoded output (lower = smaller payload for the API)
OUTPUT_JPEG_QUALITY = 82


def _try_load_font(size: int) -> ImageFont.ImageFont:
    """Attempt to load a readable monospace font; fall back to default."""
    candidates = [
        "cour.ttf",          # Courier New (Windows)
        "DejaVuSansMono.ttf",
        "LiberationMono-Regular.ttf",
        "arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def apply_grid_overlay(
    image: Image.Image,
    divisions: int = GRID_DIVISIONS,
    jpeg_quality: int = OUTPUT_JPEG_QUALITY,
) -> tuple[Image.Image, str]:
    """
    Draw a semi-transparent numbered coordinate grid on top of *image*.

    The coordinate system used in the labels matches the VLM prompt:
      - (0, 0) = top-left
      - (1000, 1000) = bottom-right
      - grid lines are drawn at multiples of (1000 // divisions)

    Parameters
    ----------
    image:        PIL Image (any mode; will be converted to RGBA internally).
    divisions:    Number of equal divisions (default 10 → lines every 100 units).
    jpeg_quality: JPEG quality for the returned base64 string.

    Returns
    -------
    (annotated_pil, base64_str)
        annotated_pil – RGB PIL Image with grid overlay baked in.
        base64_str    – base64-encoded JPEG of the same image.
    """
    # Lấy chiều rộng (W) và chiều cao (H) tính bằng pixel của ảnh gốc.
    # Hai giá trị này dùng để quy đổi từ tọa độ chuẩn hóa [0-1000] → pixel thật.
    W, H = image.size

    # Tính khoảng cách giữa hai đường kẻ liên tiếp trên thang tọa độ 0-1000.
    # Ví dụ: divisions=10 → step=100, tức là kẻ đường tại 0, 100, 200, ..., 1000.
    # Ví dụ: divisions=100 → step=10, tức là kẻ đường tại 0, 10, 20, ..., 1000 (dày hơn).
    step = 1000 // divisions

    # Chuyển ảnh gốc sang chế độ RGBA (thêm kênh alpha = độ trong suốt).
    # Cần RGBA vì sẽ vẽ đường kẻ bán trong suốt (alpha < 255) lên overlay.
    rgba = image.convert("RGBA")

    # Tạo một lớp overlay hoàn toàn trong suốt (alpha = 0) có cùng kích thước.
    # Tất cả đường grid sẽ được vẽ lên lớp này, rồi mới blend vào ảnh gốc.
    # Kỹ thuật này giúp kiểm soát độ mờ của toàn bộ grid một cách độc lập.
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))

    # Tạo đối tượng Draw gắn vào overlay để vẽ đường lên đó.
    draw_ov = ImageDraw.Draw(overlay)

    # ── Vẽ các đường kẻ grid ───────────────────────────────────────────────────
    # Lặp qua từng giá trị tick (0, step, 2*step, ..., 1000) trên thang 0-1000.
    for tick in range(0, 1001, step):

        # Quy đổi tọa độ chuẩn hóa tick (trong [0, 1000]) → pixel thật.
        # Công thức: px = tick / 1000 * kích_thước_pixel
        # Ví dụ: tick=500, W=1000px → px_x = 500px (đúng giữa ảnh theo chiều ngang)
        px_x = int(tick * W / 1000)   # vị trí pixel theo chiều X (cột)
        px_y = int(tick * H / 1000)   # vị trí pixel theo chiều Y (hàng)

        # Tạo màu RGBA cho đường kẻ: RGB từ hằng số + kênh alpha để kiểm soát độ mờ.
        # GRID_LINE_COLOR = (160, 160, 160) — xám nhạt
        # GRID_LINE_ALPHA = 60 — khá trong suốt, không che nội dung manga
        line_color = GRID_LINE_COLOR + (GRID_LINE_ALPHA,)

        # Vẽ đường thẳng đứng (vertical) tại cột px_x, từ trên cùng (y=0) đến dưới cùng (y=H).
        # Đường này đánh dấu giá trị X = tick trên thang tọa độ 0-1000.
        draw_ov.line([(px_x, 0), (px_x, H)], fill=line_color, width=1)

        # Vẽ đường nằm ngang (horizontal) tại hàng px_y, từ trái (x=0) đến phải (x=W).
        # Đường này đánh dấu giá trị Y = tick trên thang tọa độ 0-1000.
        draw_ov.line([(0, px_y), (W, px_y)], fill=line_color, width=1)

    # Blend overlay (chứa đường kẻ bán trong suốt) vào ảnh RGBA gốc bằng alpha compositing.
    # alpha_composite xử lý kênh alpha chính xác: pixel grid hiện rõ trên nền sáng
    # nhưng vẫn để nội dung manga lộ ra qua phần trong suốt.
    # Kết quả cuối cùng convert về RGB vì không cần alpha nữa (PNG hoặc JPEG output).
    blended = Image.alpha_composite(rgba, overlay).convert("RGB")

    # Tạo đối tượng Draw mới gắn vào ảnh đã blend để vẽ nhãn chữ số lên.
    # (Không dùng lại draw_ov vì overlay đã được merge vào blended rồi.)
    draw = ImageDraw.Draw(blended)

    # ── Tính kích thước font cho nhãn số ──────────────────────────────────────
    # Lấy cạnh ngắn hơn của ảnh (tránh font quá to trên ảnh dài).
    short_side = min(W, H)

    # font_size = cạnh_ngắn / hệ_số_chia, tối thiểu là 10px để luôn đọc được.
    # LABEL_FONT_SIZE_DIVISOR = 60 → ảnh 600px → font 10px; ảnh 1200px → font 20px.
    font_size = max(10, short_side // LABEL_FONT_SIZE_DIVISOR)

    # Load font thật (monospace cho dễ đọc); nếu không có → dùng font mặc định PIL.
    font = _try_load_font(font_size)

    def _label_size(text: str):
        """Return (w, h) of the rendered label text."""
        # textbbox trả về (left, top, right, bottom) của bounding box văn bản.
        # Tính chiều rộng = right - left, chiều cao = bottom - top.
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _draw_label(x: int, y: int, text: str, anchor: str = "lt"):
        """Draw a small white-background text label at pixel (x, y)."""
        # Tính kích thước của chuỗi text sẽ render để biết cần bao nhiêu chỗ cho nền trắng.
        tw, th = _label_size(text)
        pad = LABEL_PADDING   # khoảng cách thêm xung quanh chữ (pixel)

        # Tính tọa độ góc trên-trái (bx0, by0) của nền trắng tùy theo anchor:
        # "lt" = left-top    → gốc tọa độ (x,y) là góc trên-trái của label
        # "rt" = right-top   → (x,y) là góc trên-PHẢI, nên dịch trái đi tw+pad
        # "lb" = left-bottom → (x,y) là góc dưới-trái, nên dịch lên th+pad
        # "rb" = right-bottom
        if anchor == "lt":
            bx0, by0 = x, y
        elif anchor == "rt":
            bx0, by0 = x - tw - pad, y
        elif anchor == "lb":
            bx0, by0 = x, y - th - pad
        elif anchor == "rb":
            bx0, by0 = x - tw - pad, y - th - pad
        else:
            bx0, by0 = x, y

        # Vẽ hình chữ nhật nền trắng phía sau chữ số để chữ luôn đọc được
        # dù nền manga sáng hay tối. Padding -1/+pad để có khoảng trống nhỏ quanh chữ.
        draw.rectangle(
            [bx0 - 1, by0 - 1, bx0 + tw + pad, by0 + th + pad],
            fill=LABEL_BG_COLOR,
        )

        # Vẽ chữ số lên trên nền trắng vừa tạo.
        draw.text((bx0, by0), text, fill=LABEL_TEXT_COLOR, font=font)

    # Lặp lại qua từng tick để vẽ nhãn số (lần này trên ảnh đã blend, không phải overlay).
    for tick in range(0, 1001, int(step*1)):
        label = str(tick)             # nhãn hiển thị: "0", "100", "200", ...

        # Quy đổi tọa độ chuẩn hóa → pixel (giống phần vẽ đường kẻ bên trên).
        px_x = int(tick * W / 1000)
        px_y = int(tick * H / 1000)

        # Nhãn trục X — in tại mép trên và mép dưới ảnh, ngay cạnh đường dọc.
        # Mép trên: y = LABEL_PADDING (sát trên cùng)
        # Mép dưới: y = H - font_size - LABEL_PADDING*2 (sát dưới cùng, offset để vừa)
        _draw_label(px_x + LABEL_PADDING, LABEL_PADDING, label, anchor="lt")
        _draw_label(px_x + LABEL_PADDING, H - font_size - LABEL_PADDING * 2, label, anchor="lt")

        # Nhãn trục Y — in tại mép trái và mép phải ảnh, ngay cạnh đường ngang.
        # Bỏ qua tick=0 vì nhãn "0" của Y sẽ chồng lên nhãn "0" của X (đã in ở trên).
        if tick > 0:
            # Mép trái: x = LABEL_PADDING, anchor lt (căn trái)
            _draw_label(LABEL_PADDING, px_y + LABEL_PADDING, label, anchor="lt")
            # Mép phải: x = W - LABEL_PADDING - 1, anchor rt (căn phải để không tràn ra ngoài)
            _draw_label(W - LABEL_PADDING - 1, px_y + LABEL_PADDING, label, anchor="rt")

    # ── Encode ảnh đã annotate thành JPEG base64 ──────────────────────────────
    # Tạo buffer RAM thay vì ghi ra file disk — nhanh hơn và không cần cleanup.
    buffer = io.BytesIO()

    # Lưu ảnh RGB vào buffer theo định dạng JPEG với quality tùy chỉnh.
    # JPEG nhỏ hơn PNG nhiều → payload gửi lên Gemini API nhỏ hơn → tiết kiệm token.
    blended.save(buffer, format="JPEG", quality=jpeg_quality)

    # Đọc toàn bộ bytes từ buffer → encode base64 → decode ra chuỗi Python string.
    # Gemini API nhận ảnh dưới dạng base64 string trong JSON payload.
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Log kích thước payload để dễ monitor (quá lớn → tăng compression / giảm resolution).
    logger.debug(
        f"[GridOverlay] Grid applied to {W}x{H} image "
        f"({divisions} divisions, step={step}). "
        f"Payload size: {len(b64) // 1024} KB"
    )

    # Trả về cả ảnh PIL (dùng để lưu file / hiển thị) lẫn base64 string (gửi API).
    return blended, b64

