"""
Module: app.detection.rt_detr
Description: 100% Identical PyTorch FP32 RT-DETR-v2 detection engine from comic-translate.
             Uses 'ogkalu/comic-text-and-bubble-detector' model via HuggingFace Transformers.
"""

import os
import math
import logging
import threading
import numpy as np
import cv2
from PIL import Image
from typing import Optional, List, Tuple

import imkit as imk
from app.detection.utils.slicer import ImageSlicer
from app.detection.utils.content import filter_and_fix_bboxes, adjust_text_line_coordinates
from app.detection.utils.geometry import does_rectangle_fit, do_rectangles_overlap, merge_overlapping_boxes
from app.detection.heuristic_lines.core import annotate_blocks_with_heuristic_lines
from app.detection.heuristic_lines.direction import _sort_lines

logger = logging.getLogger(__name__)

# Model HuggingFace Repository ID for PyTorch FP32
MODEL_REPO_ID = "ogkalu/comic-text-and-bubble-detector"

# Set HF_HOME cache directory inside BE/models/detection/ so PyTorch FP32 weights persist in BE/models/detection/
MODELS_DET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "detection"))
os.makedirs(MODELS_DET_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DET_DIR
os.environ["HF_HUB_CACHE"] = MODELS_DET_DIR


# ===========================================================================
# Dataclass: TextBlock (100% Compatible with comic-translate)
# ===========================================================================

class TextBlock:
    """
    Object storing a block of text, line sub-boxes, bubble box, font attributes and translations.
    Matches comic-translate's TextBlock implementation exactly.
    """
    __slots__ = (
        "xyxy", "bubble_xyxy", "text_class", "lines", "direction",
        "font_color", "text", "texts", "translation", "line_spacing",
        "alignment", "source_lang", "target_lang", "script",
        "min_font_size", "max_font_size", "angle", "tr_origin_point",
        "segm_pts", "skipped_small_texts", "bubble_id",
    )

    def __init__(
        self,
        text_bbox: np.ndarray = None,
        bubble_bbox: np.ndarray = None,
        text_class: str = "",
        lines: List = None,
        text_segm_points: np.ndarray = None,
        angle=0,
        text: str = "",
        texts: List[str] = None,
        skipped_small_texts: List[dict] = None,
        translation: str = "",
        line_spacing=1.0,
        alignment: str = "",
        source_lang: str = "",
        target_lang: str = "",
        script: str = "",
        min_font_size: int = 0,
        max_font_size: int = 0,
        font_color: str | tuple | list = (),
        direction: str = "",
        bubble_id: int | None = None,
        **kwargs,
    ):
        self.xyxy = np.array(text_bbox, dtype=int) if text_bbox is not None else None
        self.bubble_xyxy = np.array(bubble_bbox, dtype=int) if bubble_bbox is not None else None
        self.text_class = text_class  # 'text_bubble' | 'text_free'
        self.angle = angle
        self.tr_origin_point = ()
        self.segm_pts = text_segm_points

        self.lines = lines if lines is not None else []
        self.texts = texts if texts is not None else []
        self.skipped_small_texts = skipped_small_texts if skipped_small_texts is not None else []
        self.text = " ".join(self.texts) if self.texts else text
        self.translation = translation

        self.line_spacing = line_spacing
        self.alignment = alignment

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.script = script

        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.font_color = font_color
        self.direction = direction  # 'horizontal' | 'vertical'
        self.bubble_id = bubble_id

    @property
    def xywh(self) -> np.ndarray:
        if self.xyxy is None:
            return np.array([0, 0, 0, 0], dtype=np.int32)
        x1, y1, x2, y2 = self.xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.int32)

    @property
    def center(self) -> np.ndarray:
        if self.xyxy is None:
            return np.array([0.0, 0.0])
        xyxy = np.array(self.xyxy, dtype=float)
        return (xyxy[:2] + xyxy[2:]) / 2.0

    @property
    def source_lang_direction(self) -> str:
        return "ver_rtl" if self.direction == "vertical" else "hor_ltr"

    def __repr__(self):
        return (
            f"TextBlock(class={self.text_class!r}, xyxy={list(self.xyxy) if self.xyxy is not None else None}, "
            f"bubble={list(self.bubble_xyxy) if self.bubble_xyxy is not None else None}, "
            f"lines={len(self.lines)}, direction={self.direction!r}, color={self.font_color})"
        )


# ===========================================================================
# Border-Otsu Foreground Color Extraction (100% comic-translate algorithm)
# ===========================================================================

def snap_extreme_neutrals(rgb: list) -> list:
    """Snap chromatic neutrals to pure black [0,0,0] or pure white [255,255,255]."""
    r, g, b = rgb
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = max(r, g, b) - min(r, g, b)
    if chroma < 40:
        return [0, 0, 0] if luma < 128 else [255, 255, 255]
    return [r, g, b]


def extract_foreground_color(crop_rgb: np.ndarray) -> list | None:
    """
    Extract text color using Border-Otsu Euclidean distance thresholding.
    100% identical to comic-translate implementation.
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return None

    h, w = crop_rgb.shape[:2]
    if h < 6 or w < 6:
        return None

    img = crop_rgb[:, :, :3].astype(np.float64)
    bw = max(2, min(h, w) // 8)
    if h <= bw * 2 or w <= bw * 2:
        return None

    # Sample 4 border strips to estimate background color
    top    = img[:bw, bw:-bw] if w > bw * 2 else img[:bw, :]
    bottom = img[-bw:, bw:-bw] if w > bw * 2 else img[-bw:, :]
    left   = img[bw:-bw, :bw]
    right  = img[bw:-bw, -bw:]

    border_parts = [p.reshape(-1, 3) for p in (top, bottom, left, right) if p.size > 0]
    if not border_parts:
        return None

    border_pixels = np.concatenate(border_parts, axis=0)
    bg = np.median(border_pixels, axis=0)

    # Compute Euclidean distance map to background color
    flat = img.reshape(-1, 3)
    dist = np.sqrt(np.sum((flat - bg) ** 2, axis=1))

    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(otsu_thresh), 25.0)

    text_mask = dist > threshold
    if int(np.sum(text_mask)) < 5:
        return None

    text_pixels = flat[text_mask]
    bg_luma = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]

    if bg_luma >= 170:
        fg = np.percentile(text_pixels, 20, axis=0)
    elif bg_luma <= 85:
        fg = np.percentile(text_pixels, 80, axis=0)
    else:
        fg = np.median(text_pixels, axis=0)

    return snap_extreme_neutrals(np.round(fg).astype(int).tolist())


# ===========================================================================
# RT-DETR-v2 PyTorch Detector Engine (FP32)
# ===========================================================================

class RTDetrV2PyTorchDetector:
    """
    100% PyTorch FP32 detection engine matching comic-translate's RTDetrV2Detection.
    Runs 'ogkalu/comic-text-and-bubble-detector' model via HuggingFace Transformers.
    """

    def __init__(
        self,
        model_path_or_repo: str = MODEL_REPO_ID,
        confidence_threshold: float = 0.30,
        device: str | None = None,
    ):
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        # Set HF_HOME cache directory inside BE/models/detection/ so PyTorch FP32 weights persist in BE/models/detection/
        models_det_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "detection"))
        os.makedirs(models_det_dir, exist_ok=True)
        os.environ["HF_HOME"] = models_det_dir

        self.confidence_threshold = confidence_threshold
        self.repo_name = model_path_or_repo

        # Resolve device: GPU (cuda) if available, else CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"[Detector FP32] Loading PyTorch model from '{self.repo_name}' on device '{self.device}'...")

        self.processor = RTDetrImageProcessor.from_pretrained(
            self.repo_name,
            size={"width": 640, "height": 640},
        )
        self.model = RTDetrV2ForObjectDetection.from_pretrained(
            self.repo_name,
        ).to(self.device)
        self.model.eval()

        self.image_slicer = ImageSlicer(
            height_to_width_ratio_threshold=3.5,
            target_slice_ratio=3.0,
            overlap_height_ratio=0.2,
            min_slice_height_ratio=0.7,
        )

        logger.info(f"[Detector FP32] RT-DETR-v2 PyTorch FP32 model initialized successfully.")

    def detect(self, image: np.ndarray) -> list[TextBlock]:
        """
        Full detection pipeline:
          1. ImageSlicer for tall webtoon images.
          2. _detect_single_image (PyTorch FP32 inference).
          3. create_text_blocks (filtering, merging, color extraction, bubble matching).
          4. annotate_blocks_with_heuristic_lines (heuristic line segmentation & direction).
          5. Assign sequential bubble_id (1, 2, 3...).
        """
        bubble_boxes, text_boxes = self.image_slicer.process_slices_for_detection(
            image, self._detect_single_image
        )
        text_blocks = self.create_text_blocks(image, text_boxes, bubble_boxes)

        # Sort blocks into manga reading order
        text_blocks = sort_text_blocks_manga_order(text_blocks)

        # Assign bubble_id (1, 2, 3...)
        for idx, blk in enumerate(text_blocks, start=1):
            blk.bubble_id = idx

        return text_blocks

    def _detect_single_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runs PyTorch FP32 inference on a single image array (RGB)."""
        import torch

        pil_image = Image.fromarray(image)  # RGB PIL image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )[0]

        bubble_boxes = []
        text_boxes = []

        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            box_coords = [int(v) for v in box.tolist()]
            label_id = int(label.item())

            # Class 0: bubble, Class 1/2: text_bubble / text_free
            if label_id == 0:
                bubble_boxes.append(box_coords)
            elif label_id in (1, 2):
                text_boxes.append(box_coords)

        return (
            np.array(bubble_boxes, dtype=int) if bubble_boxes else np.empty((0, 4), dtype=int),
            np.array(text_boxes, dtype=int) if text_boxes else np.empty((0, 4), dtype=int),
        )

    def create_text_blocks(
        self,
        image: np.ndarray,
        text_boxes: np.ndarray,
        bubble_boxes: Optional[np.ndarray] = None,
    ) -> list[TextBlock]:
        """
        Creates TextBlock objects, extracts foreground color, matches bubbles,
        and annotates heuristic text lines. 100% matches comic-translate's base.py.
        """
        h, w = image.shape[:2]
        text_boxes = filter_and_fix_bboxes(text_boxes, (h, w))
        bubble_boxes = filter_and_fix_bboxes(bubble_boxes, (h, w)) if bubble_boxes is not None else np.empty((0, 4), dtype=int)
        text_boxes = merge_overlapping_boxes(text_boxes)

        if len(text_boxes) == 0:
            return []

        # 1. Extract foreground color via Border-Otsu
        text_colors_per_box: list[tuple] = [()] * len(text_boxes)
        for txt_idx, txt_box in enumerate(text_boxes):
            x1, y1, x2, y2 = map(int, txt_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                color = extract_foreground_color(image[y1:y2, x1:x2])
                if color is not None:
                    text_colors_per_box[txt_idx] = tuple(color)

        # 2. Build TextBlock objects and match with bubble boxes
        text_blocks: list[TextBlock] = []
        text_matched = [False] * len(text_boxes)

        for txt_idx, txt_box in enumerate(text_boxes):
            text_color = text_colors_per_box[txt_idx]

            if len(bubble_boxes) == 0:
                text_blocks.append(
                    TextBlock(
                        text_bbox=txt_box,
                        text_class="text_free",
                        font_color=text_color,
                    )
                )
                continue

            for bble_box in bubble_boxes:
                if bble_box is None:
                    continue
                if does_rectangle_fit(bble_box, txt_box) or do_rectangles_overlap(bble_box, txt_box):
                    text_blocks.append(
                        TextBlock(
                            text_bbox=txt_box,
                            bubble_bbox=bble_box,
                            text_class="text_bubble",
                            font_color=text_color,
                        )
                    )
                    text_matched[txt_idx] = True
                    break

            if not text_matched[txt_idx]:
                text_blocks.append(
                    TextBlock(
                        text_bbox=txt_box,
                        text_class="text_free",
                        font_color=text_color,
                    )
                )

        # 3. Annotate text blocks with heuristic lines & direction
        try:
            annotate_blocks_with_heuristic_lines(image, text_blocks)
        except Exception as e:
            logger.warning(f"Failed to build heuristic text lines: {e}")

        return text_blocks


# ===========================================================================
# Reading Order Sorting Helper
# ===========================================================================

def sort_text_blocks_manga_order(blk_list: list[TextBlock], right_to_left: bool = True) -> list[TextBlock]:
    """Sorts TextBlock list into reading order based on Y & X center coordinates."""
    if not blk_list:
        return blk_list

    sorted_blk_list = []
    for blk in sorted(blk_list, key=lambda b: b.center[1]):
        for i, sorted_blk in enumerate(sorted_blk_list):
            if blk.center[1] > sorted_blk.xyxy[3]:
                continue
            if blk.center[1] < sorted_blk.xyxy[1]:
                sorted_blk_list.insert(i + 1, blk)
                break

            pair_is_vertical = (blk.direction == "vertical") or (sorted_blk.direction == "vertical")
            pair_rtl = right_to_left if pair_is_vertical else False

            if pair_rtl and blk.center[0] > sorted_blk.center[0]:
                sorted_blk_list.insert(i, blk)
                break
            if not pair_rtl and blk.center[0] < sorted_blk.center[0]:
                sorted_blk_list.insert(i, blk)
                break
        else:
            sorted_blk_list.append(blk)

    return sorted_blk_list


# ===========================================================================
# Singleton Factory (Strictly PyTorch FP32 per user instruction)
# ===========================================================================

_detector_instance: RTDetrV2PyTorchDetector | None = None
_detector_lock = threading.Lock()


def get_detector() -> RTDetrV2PyTorchDetector:
    """
    Returns singleton RTDetrV2PyTorchDetector instance.
    Strictly PyTorch FP32 without fallback per user directive.
    """
    global _detector_instance

    if _detector_instance is not None:
        return _detector_instance

    with _detector_lock:
        if _detector_instance is not None:
            return _detector_instance

        from app.core.config import settings

        model_path = getattr(settings, "DETECTOR_MODEL_PATH", MODEL_REPO_ID)
        conf_thresh = getattr(settings, "DETECTOR_CONF_THRESH", 0.30)

        # Check if local model directory exists
        if os.path.exists(model_path):
            repo_or_path = model_path
        else:
            repo_or_path = MODEL_REPO_ID

        _detector_instance = RTDetrV2PyTorchDetector(
            model_path_or_repo=repo_or_path,
            confidence_threshold=conf_thresh,
        )

    return _detector_instance
