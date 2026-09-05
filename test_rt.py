#!/usr/bin/env python3
"""
Script: test_rt.py
Description: Visual test script for RT-DETR-v2 PyTorch FP32 text block and speech bubble detection.

Usage:
    python test_rt.py --image path/to/manga_page.jpg --conf 0.30 --output result.jpg

Options:
    -i, --image   Path to input manga/webtoon image (required).
    -c, --conf    Confidence score threshold (default: 0.30).
    -o, --output  Path to save annotated output image (default: output_detected.jpg).
"""

import sys
import os
import argparse
import time
import numpy as np
import cv2

# Ensure BE/ directory is in Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BE"))

from app.detection.rt_detr import RTDetrV2PyTorchDetector, MODEL_REPO_ID, TextBlock


def visualize_detections(img_rgb: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
    """
    Visualizes detected text blocks and speech bubbles on the image:
      - GREEN box: Speech Bubble bbox (bubble_xyxy)
      - RED box: Tight Text BBox (xyxy)
      - CYAN boxes/polygons: Sub-line segmentation (lines) from heuristic_lines
      - RED badge: bubble_id (1, 2, 3...) at top-left of xyxy
    """
    canvas = img_rgb.copy()
    h_img, w_img = canvas.shape[:2]

    line_thickness = max(2, int(round(min(h_img, w_img) / 500)))

    for blk in blocks:
        bid = getattr(blk, "bubble_id", 0)

        # 1. Draw Speech Bubble Box (GREEN) if present (no text label on bubble)
        if blk.bubble_xyxy is not None:
            bx1, by1, bx2, by2 = map(int, blk.bubble_xyxy)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 0), line_thickness)

        # 2. Draw Tight Text Box xyxy (RED)
        if blk.xyxy is not None:
            x1, y1, x2, y2 = map(int, blk.xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), line_thickness)

            # 3. Draw Sub-lines from heuristic_lines (YELLOW/CYAN)
            for line_box in getattr(blk, "lines", []):
                if line_box is None or len(line_box) < 4:
                    continue
                l_arr = np.array(line_box, dtype=int)
                if l_arr.ndim == 1 and len(l_arr) >= 4:
                    lx1, ly1, lx2, ly2 = l_arr[:4]
                    cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (255, 200, 0), max(1, line_thickness - 1))
                elif l_arr.ndim == 2 and len(l_arr) >= 3:
                    cv2.polylines(canvas, [l_arr], isClosed=True, color=(255, 200, 0), thickness=max(1, line_thickness - 1))

            # 4. Draw bubble_id badge (White background + Red text) ABOVE top-left of xyxy
            label_str = f"#{bid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            txt_thick = 2
            (tw, th), baseline = cv2.getTextSize(label_str, font, font_scale, txt_thick)

            badge_w = tw + 8
            badge_h = th + 8

            badge_x1 = max(0, x1)
            badge_x2 = min(w_img, badge_x1 + badge_w)

            # Position badge above y1, or fallback to y1 if near top edge of image
            if y1 - badge_h >= 0:
                badge_y2 = y1 - 2
                badge_y1 = badge_y2 - badge_h
            else:
                badge_y1 = max(0, y1)
                badge_y2 = badge_y1 + badge_h

            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), -1)
            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (220, 0, 0), 1)
            cv2.putText(
                canvas, label_str,
                (badge_x1 + 4, badge_y2 - 4),
                font, font_scale, (220, 0, 0), txt_thick, cv2.LINE_AA
            )

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Test RT-DETR-v2 PyTorch FP32 text block and speech bubble detection."
    )
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input manga/webtoon image")
    parser.add_argument("-c", "--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("-o", "--output", type=str, default="output_detected.jpg", help="Output annotated image path")
    parser.add_argument("--model", type=str, default=MODEL_REPO_ID, help="Model path or HF repo ID")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ ERROR: File ảnh không tồn tại: '{args.image}'")
        sys.exit(1)

    print("=" * 70)
    print(" 🚀 MANGA TEXT BLOCK DETECTOR (PyTorch FP32 Test)")
    print("=" * 70)
    print(f" 📄 Input Image : {args.image}")
    print(f" 🎯 Conf Thresh : {args.conf}")
    print(f" 🤖 Model Path  : {args.model}")
    print(f" 💾 Output Path : {args.output}")
    print("=" * 70)

    # 1. Load image
    bgr_img = cv2.imread(args.image)
    if bgr_img is None:
        print(f"❌ ERROR: Không thể đọc file ảnh bằng OpenCV: '{args.image}'")
        sys.exit(1)

    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"📸 Kích thước ảnh: {w} × {h} px")

    # 2. Initialize FP32 PyTorch Detector
    print("\n⏳ Đang khởi tạo RT-DETR-v2 PyTorch FP32 model...")
    t0 = time.perf_counter()
    detector = RTDetrV2PyTorchDetector(
        model_path_or_repo=args.model,
        confidence_threshold=args.conf,
    )
    t_load = time.perf_counter() - t0
    print(f"✅ Khởi tạo model thành công trong {t_load:.2f}s!")

    # 3. Run Detection
    print("\n🔍 Đang chạy detection + bóc tách dòng chữ (heuristic_lines)...")
    t1 = time.perf_counter()
    blocks = detector.detect(img_rgb)
    t_detect = time.perf_counter() - t1

    print(f"✅ Chạy detection xong trong {t_detect:.2f}s! Phát hiện {len(blocks)} khối chữ (TextBlock).\n")

    # 4. Print detailed block stats
    print("-" * 70)
    print(f"{'ID':<5} | {'Class':<12} | {'Direction':<10} | {'Lines':<6} | {'Font Color (RGB)':<18} | {'Text BBox (xyxy)'}")
    print("-" * 70)

    for blk in blocks:
        bid = getattr(blk, "bubble_id", 0)
        cls_name = blk.text_class or "text_free"
        direction = blk.direction or "horizontal"
        num_lines = len(getattr(blk, "lines", []))
        color_str = str(list(blk.font_color)) if blk.font_color else "None"
        xyxy_str = str(list(blk.xyxy)) if blk.xyxy is not None else "None"
        print(f"#{bid:<4} | {cls_name:<12} | {direction:<10} | {num_lines:<6} | {color_str:<18} | {xyxy_str}")

    print("-" * 70)

    # 5. Visualize and Save output image
    annotated_rgb = visualize_detections(img_rgb, blocks)
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(args.output, annotated_bgr)
    print(f"\n🎉 Ảnh kết quả khoanh vùng đã được lưu tại: '{args.output}'")
    print("  - Khung XANH LÁ : Bong bóng thoại (bubble_xyxy)")
    print("  - Khung ĐỎ      : Vùng chữ thắt chặt (xyxy)")
    print("  - Khung VÀNG/CYAN: Các dòng chữ bóc tách (lines)")
    print("  - Thẻ ĐỎ        : bubble_id (1, 2, 3...)")
    print("=" * 70)


if __name__ == "__main__":
    main()
