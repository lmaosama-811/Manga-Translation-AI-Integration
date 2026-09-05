"""
Module: app.services.inpainting
Description: Comic-translate 1:1 Inpainting Pipeline.
             Implements generate_mask(), _apply_fast_bubble_cleanup(), 
             and inpaint_image() (hybrid fast-fill + NN LaMa inpainting).
"""

import cv2
import asyncio
import numpy as np
import torch
import logging
import imkit as imk

from app.services.inpainting_boxes import merge_overlapping_padded_boxes
from app.services.content_detection import detect_content_mask_in_bbox, adjust_text_line_coordinates

from manga_translator.config import Config
from manga_translator.inpainting import dispatch as dispatch_inpainting, get_inpainter
from manga_translator.inpainting.common import OfflineInpainter

logger = logging.getLogger(__name__)

# Constants from comic-translate
FAST_FILL_BUBBLE_INSET = 7
FAST_FILL_UNIFORM_COLOR_TOLERANCE = 12
FAST_FILL_MIN_UNIFORM_COVERAGE = 0.94

# ------------------------------------------------------------------
# Global LaMa Semaphore
# ------------------------------------------------------------------

_LAMA_GLOBAL_SEM: asyncio.Semaphore | None = None

def _get_lama_semaphore() -> asyncio.Semaphore:
    global _LAMA_GLOBAL_SEM
    if _LAMA_GLOBAL_SEM is None:
        _LAMA_GLOBAL_SEM = asyncio.Semaphore(1)
        logger.info("[LaMa] Global semaphore(1) initialized — inpainting serialized across all requests.")
    return _LAMA_GLOBAL_SEM


# ==========================================================================
# COPY-PASTE from comic-translate/modules/utils/image_utils.py
# ==========================================================================

def build_bubble_clip_mask(
    mask_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
    bubble_xyxy,
    *,
    inset: int,
    image: np.ndarray | None = None,
    seed_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    if bubble_xyxy is None or len(bubble_xyxy) < 4:
        return None

    x1, y1, x2, y2 = [int(v) for v in bounds]
    bx1, by1, bx2, by2 = [int(v) for v in bubble_xyxy[:4]]

    bx1_rel = bx1 + inset - x1
    by1_rel = by1 + inset - y1
    bx2_rel = bx2 - inset - x1
    by2_rel = by2 - inset - y1

    height, width = mask_shape[:2]
    use_fallback = True

    if image is not None:
        try:
            H, W = image.shape[:2]
            margin = 5
            crop_y1 = max(0, by1 - margin)
            crop_y2 = min(H, by2 + margin)
            crop_x1 = max(0, bx1 - margin)
            crop_x2 = min(W, bx2 + margin)

            bubble_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

            if bubble_crop.ndim == 3:
                gray = (0.299 * bubble_crop[..., 2] + 0.587 * bubble_crop[..., 1] + 0.114 * bubble_crop[..., 0]).astype(np.uint8)
            else:
                gray = bubble_crop.copy()

            if seed_bbox is not None:
                sx1, sy1, sx2, sy2 = [int(v) for v in seed_bbox[:4]]
            else:
                sx1 = (bx1 + bx2) // 2 - 5
                sx2 = (bx1 + bx2) // 2 + 5
                sy1 = (by1 + by2) // 2 - 5
                sy2 = (by1 + by2) // 2 + 5

            seed_y1_rel = max(0, sy1 - crop_y1)
            seed_y2_rel = min(crop_y2 - crop_y1, sy2 - crop_y1)
            seed_x1_rel = max(0, sx1 - crop_x1)
            seed_x2_rel = min(crop_x2 - crop_x1, sx2 - crop_x1)

            seed_region = gray[seed_y1_rel:seed_y2_rel, seed_x1_rel:seed_x2_rel]

            if seed_region.size > 0:
                hist, bin_edges = np.histogram(seed_region, bins=16, range=(0, 256))
                max_bin = np.argmax(hist)
                bg_val = (bin_edges[max_bin] + bin_edges[max_bin+1]) / 2.0

                tolerance = 20
                bg_mask = np.abs(gray - bg_val) <= tolerance

                num_labels, labeled = imk.connected_components(bg_mask, connectivity=4)

                seed_pixels_mask = bg_mask[seed_y1_rel:seed_y2_rel, seed_x1_rel:seed_x2_rel]
                seed_labels = labeled[seed_y1_rel:seed_y2_rel, seed_x1_rel:seed_x2_rel][seed_pixels_mask]
                unique_labels = np.unique(seed_labels)
                unique_labels = unique_labels[unique_labels > 0]

                if unique_labels.size > 0:
                    bubble_mask = np.isin(labeled, unique_labels)

                    b_y1_rel = by1 - crop_y1
                    b_y2_rel = by2 - crop_y1
                    b_x1_rel = bx1 - crop_x1
                    b_x2_rel = bx2 - crop_x1

                    border_mask_pixels = []
                    if 0 <= b_y1_rel < bubble_mask.shape[0]:
                        border_mask_pixels.extend(bubble_mask[b_y1_rel, max(0, b_x1_rel):min(bubble_mask.shape[1], b_x2_rel)])
                    if 0 <= b_y2_rel - 1 < bubble_mask.shape[0]:
                        border_mask_pixels.extend(bubble_mask[b_y2_rel - 1, max(0, b_x1_rel):min(bubble_mask.shape[1], b_x2_rel)])
                    if 0 <= b_x1_rel < bubble_mask.shape[1]:
                        border_mask_pixels.extend(bubble_mask[max(0, b_y1_rel):min(bubble_mask.shape[0], b_y2_rel), b_x1_rel])
                    if 0 <= b_x2_rel - 1 < bubble_mask.shape[1]:
                        border_mask_pixels.extend(bubble_mask[max(0, b_y1_rel):min(bubble_mask.shape[0], b_y2_rel), b_x2_rel - 1])

                    border_mask_pixels = np.array(border_mask_pixels)
                    touch_ratio = np.mean(border_mask_pixels) if border_mask_pixels.size > 0 else 0.0

                    if touch_ratio < 0.5:
                        use_fallback = False

                    if not use_fallback:
                        bubble_mask = imk.close_holes(bubble_mask)
                        seg_inset = min(2, inset)
                        if seg_inset > 0:
                            struct_elem = imk.get_structuring_element(imk.MORPH_CROSS, (3, 3))
                            bubble_mask = imk.erode(bubble_mask.astype(np.uint8) * 255, struct_elem, iterations=seg_inset) > 0

                        final_clip = np.zeros(mask_shape, dtype=bool)
                        overlap_y1 = max(y1, crop_y1)
                        overlap_y2 = min(y2, crop_y2)
                        overlap_x1 = max(x1, crop_x1)
                        overlap_x2 = min(x2, crop_x2)

                        if overlap_y2 > overlap_y1 and overlap_x2 > overlap_x1:
                            f_y1 = overlap_y1 - y1
                            f_y2 = overlap_y2 - y1
                            f_x1 = overlap_x1 - x1
                            f_x2 = overlap_x2 - x1

                            b_y1 = overlap_y1 - crop_y1
                            b_y2 = overlap_y2 - crop_y1
                            b_x1 = overlap_x1 - crop_x1
                            b_x2 = overlap_x2 - crop_x1

                            final_clip[f_y1:f_y2, f_x1:f_x2] = bubble_mask[b_y1:b_y2, b_x1:b_x2]

                        cy_grid2, cx_grid2 = np.ogrid[:height, :width]
                        ellipse_cx2 = (bx1_rel + bx2_rel) / 2.0
                        ellipse_cy2 = (by1_rel + by2_rel) / 2.0
                        rx2 = max(1.0, (bx2_rel - bx1_rel) / 2.0)
                        ry2 = max(1.0, (by2_rel - by1_rel) / 2.0)
                        ellipse_clip = (((cx_grid2 - ellipse_cx2) / rx2) ** 2 + ((cy_grid2 - ellipse_cy2) / ry2) ** 2) <= 1.0

                        if seed_bbox is not None:
                            seed_padding = 2
                            seed_x1 = max(0, sx1 - seed_padding - x1)
                            seed_y1 = max(0, sy1 - seed_padding - y1)
                            seed_x2 = min(width, sx2 + seed_padding - x1)
                            seed_y2 = min(height, sy2 + seed_padding - y1)
                            seed_envelope = np.zeros(mask_shape, dtype=bool)
                            if seed_x2 > seed_x1 and seed_y2 > seed_y1:
                                seed_envelope[seed_y1:seed_y2, seed_x1:seed_x2] = True
                                return final_clip | (ellipse_clip & seed_envelope)

                        return final_clip
        except Exception:
            pass

    cy_grid, cx_grid = np.ogrid[:height, :width]
    ellipse_cx = (bx1_rel + bx2_rel) / 2.0
    ellipse_cy = (by1_rel + by2_rel) / 2.0
    rx = max(1.0, (bx2_rel - bx1_rel) / 2.0)
    ry = max(1.0, (by2_rel - by1_rel) / 2.0)
    return (((cx_grid - ellipse_cx) / rx) ** 2 + ((cy_grid - ellipse_cy) / ry) ** 2) <= 1.0


def clip_mask_to_bubble(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    bubble_xyxy,
    *,
    inset: int,
    image: np.ndarray | None = None,
    seed_bbox=None,
) -> np.ndarray:
    bubble_clip = build_bubble_clip_mask(
        mask.shape[:2],
        bounds,
        bubble_xyxy,
        inset=inset,
        image=image,
        seed_bbox=seed_bbox,
    )
    if bubble_clip is None:
        return mask
    return np.where(bubble_clip, mask, 0).astype(mask.dtype, copy=False)


def clip_mask_components_to_bubble(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    bubble_xyxy,
    *,
    inset: int,
    image: np.ndarray | None = None,
    seed_bbox=None,
    dilate_kernel_size: int = 0,
    dilate_iterations: int = 1,
) -> np.ndarray:
    bubble_clip = build_bubble_clip_mask(
        mask.shape[:2],
        bounds,
        bubble_xyxy,
        inset=inset,
        image=image,
        seed_bbox=seed_bbox,
    )
    if bubble_clip is None:
        if dilate_kernel_size > 0:
            dil_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            return imk.dilate(mask, dil_kernel, iterations=dilate_iterations)
        return mask

    num_labels, labeled_text = imk.connected_components(mask > 0, connectivity=4)
    overlapping_labels = np.unique(labeled_text[bubble_clip])
    keep_labels = overlapping_labels[overlapping_labels > 0]

    if keep_labels.size == 0:
        return np.zeros_like(mask)

    kept_mask = np.isin(labeled_text, keep_labels)
    kept_mask_clipped = kept_mask & bubble_clip

    if dilate_kernel_size > 0:
        dil_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        dilated = imk.dilate(kept_mask_clipped.astype(np.uint8) * 255, dil_kernel, iterations=dilate_iterations)
        final_mask = np.where(bubble_clip, dilated, 0).astype(np.uint8)
        return np.bitwise_or(final_mask, (kept_mask_clipped * 255).astype(np.uint8)).astype(mask.dtype, copy=False)
    else:
        return (kept_mask_clipped * 255).astype(mask.dtype, copy=False)


def _resolve_block_crop_bounds(
    img: np.ndarray,
    blk,
    default_padding: int,
) -> tuple[int, int, int, int]:
    cx1, cy1, cx2, cy2 = adjust_text_line_coordinates(blk.xyxy, 10, 10, img)
    bubble_xyxy = getattr(blk, "bubble_xyxy", None)
    if getattr(blk, "text_class", None) != "text_bubble" or bubble_xyxy is None or len(bubble_xyxy) < 4:
        return cx1, cy1, cx2, cy2

    bx1, by1, bx2, by2 = [int(v) for v in bubble_xyxy[:4]]
    bubble_margin = max(4, min(default_padding + 3, 12))
    bubble_inset_y = max(2, min(default_padding + 1, 8))

    cx1 = min(cx1, max(0, bx1 - bubble_margin))
    cy1 = min(cy1, max(0, by1 + bubble_inset_y))
    cx2 = max(cx2, min(img.shape[1], bx2 + bubble_margin))
    cy2 = max(cy2, min(img.shape[0], by2 - bubble_inset_y))
    return cx1, cy1, cx2, cy2


def _select_text_like_components(
    crop_mask: np.ndarray,
    text_bounds: tuple[int, int, int, int],
    *,
    search_padding: int,
    core_padding: int = 2,
) -> np.ndarray:
    binary = crop_mask > 0
    num_labels, labels, stats, _centroids = imk.connected_components_with_stats(
        binary, connectivity=4
    )
    if num_labels <= 1:
        return np.zeros(crop_mask.shape[:2], dtype=np.uint8)

    height, width = crop_mask.shape[:2]
    tx1, ty1, tx2, ty2 = [int(v) for v in text_bounds]
    search_region = np.zeros((height, width), dtype=bool)
    sx1 = max(0, tx1 - search_padding)
    sy1 = max(0, ty1 - search_padding)
    sx2 = min(width, tx2 + search_padding)
    sy2 = min(height, ty2 + search_padding)
    if sx2 <= sx1 or sy2 <= sy1:
        return np.zeros(crop_mask.shape[:2], dtype=np.uint8)
    search_region[sy1:sy2, sx1:sx2] = True

    candidate_labels = np.unique(labels[search_region])
    candidate_labels = candidate_labels[candidate_labels > 0]
    if candidate_labels.size == 0:
        return np.zeros(crop_mask.shape[:2], dtype=np.uint8)

    core_region = np.zeros((height, width), dtype=bool)
    cx1 = max(0, tx1 - core_padding)
    cy1 = max(0, ty1 - core_padding)
    cx2 = min(width, tx2 + core_padding)
    cy2 = min(height, ty2 + core_padding)
    core_region[cy1:cy2, cx1:cx2] = True
    core_counts = np.bincount(labels[core_region].ravel(), minlength=num_labels)

    text_width = max(1, tx2 - tx1)
    text_height = max(1, ty2 - ty1)
    outline_length_threshold = max(24, int(round(0.30 * max(text_width, text_height))))

    def is_probable_outline(label: int) -> bool:
        component_width = int(stats[label, imk.CC_STAT_WIDTH])
        component_height = int(stats[label, imk.CC_STAT_HEIGHT])
        component_area = int(stats[label, imk.CC_STAT_AREA])
        bbox_area = max(1, component_width * component_height)
        density = component_area / float(bbox_area)
        return density < 0.08 and max(component_width, component_height) > outline_length_threshold

    anchor_labels = []
    for label_value in candidate_labels:
        label = int(label_value)
        area = max(1, int(stats[label, imk.CC_STAT_AREA]))
        core_coverage = int(core_counts[label]) / float(area)
        if core_coverage >= 0.20 and not is_probable_outline(label):
            anchor_labels.append(label)

    if not anchor_labels:
        return np.zeros(crop_mask.shape[:2], dtype=np.uint8)

    keep_labels = set(anchor_labels)
    for label_value in candidate_labels:
        label = int(label_value)
        if label in keep_labels or is_probable_outline(label):
            continue

        x = int(stats[label, imk.CC_STAT_LEFT])
        y = int(stats[label, imk.CC_STAT_TOP])
        component_width = int(stats[label, imk.CC_STAT_WIDTH])
        component_height = int(stats[label, imk.CC_STAT_HEIGHT])
        area = max(1, int(stats[label, imk.CC_STAT_AREA]))
        x2 = x + component_width
        y2 = y + component_height

        for anchor in anchor_labels:
            anchor_x = int(stats[anchor, imk.CC_STAT_LEFT])
            anchor_y = int(stats[anchor, imk.CC_STAT_TOP])
            anchor_width = int(stats[anchor, imk.CC_STAT_WIDTH])
            anchor_height = int(stats[anchor, imk.CC_STAT_HEIGHT])
            anchor_area = max(1, int(stats[anchor, imk.CC_STAT_AREA]))
            anchor_x2 = anchor_x + anchor_width
            anchor_y2 = anchor_y + anchor_height

            area_similarity = min(area, anchor_area) / float(max(area, anchor_area))
            height_similarity = min(component_height, anchor_height) / float(max(component_height, anchor_height))
            width_similarity = min(component_width, anchor_width) / float(max(component_width, anchor_width))
            row_overlap = max(0, min(y2, anchor_y2) - max(y, anchor_y))
            column_overlap = max(0, min(x2, anchor_x2) - max(x, anchor_x))
            horizontal_gap = max(0, max(x, anchor_x) - min(x2, anchor_x2))
            vertical_gap = max(0, max(y, anchor_y) - min(y2, anchor_y2))

            same_row = (
                row_overlap >= 0.50 * min(component_height, anchor_height)
                and height_similarity >= 0.45
                and horizontal_gap <= max(18, int(round(0.80 * max(component_height, anchor_height))))
            )
            same_column = (
                column_overlap >= 0.50 * min(component_width, anchor_width)
                and width_similarity >= 0.45
                and vertical_gap <= max(18, int(round(0.80 * max(component_width, anchor_width))))
            )
            if area_similarity >= 0.18 and (same_row or same_column):
                keep_labels.add(label)
                break

    return np.where(np.isin(labels, list(keep_labels)), 255, 0).astype(np.uint8)


def build_block_mask_data(
    img: np.ndarray,
    blk,
    default_padding: int = 5,
    require_text_or_translation: bool = True,
    clip_to_bubble: bool = False,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    if require_text_or_translation and not getattr(blk, 'translation', None) and not getattr(blk, 'text', None):
        return None, None

    cx1, cy1, cx2, cy2 = _resolve_block_crop_bounds(img, blk, default_padding)
    crop = img[cy1:cy2, cx1:cx2]

    crop_mask = detect_content_mask_in_bbox(crop)
    if crop_mask is None or not np.any(crop_mask):
        return None, None

    close_kernel = imk.get_structuring_element(imk.MORPH_RECT, (3, 3))
    crop_mask = imk.morphology_ex(crop_mask, imk.MORPH_CLOSE, close_kernel)

    if clip_to_bubble and getattr(blk, "text_class", None) == "text_bubble" and getattr(blk, "bubble_xyxy", None) is not None:
        tx1, ty1, tx2, ty2 = [int(round(float(v))) for v in blk.xyxy[:4]]
        text_bounds = (tx1 - cx1, ty1 - cy1, tx2 - cx1, ty2 - cy1)
        search_padding = max(16, min(default_padding + 23, 32))
        crop_mask = _select_text_like_components(
            crop_mask,
            text_bounds,
            search_padding=search_padding,
        )
    kernel_size = default_padding
    dilate_iterations = 3

    if clip_to_bubble and getattr(blk, "text_class", None) == "text_bubble" and getattr(blk, "bubble_xyxy", None) is not None:
        inset = max(1, kernel_size)
        dilated_crop_mask = clip_mask_components_to_bubble(
            crop_mask,
            (cx1, cy1, cx2, cy2),
            blk.bubble_xyxy,
            inset=inset,
            image=img,
            seed_bbox=blk.xyxy,
            dilate_kernel_size=kernel_size,
            dilate_iterations=dilate_iterations,
        )
    else:
        dil_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_crop_mask = imk.dilate(crop_mask, dil_kernel, iterations=dilate_iterations)

    if clip_to_bubble and getattr(blk, "text_class", None) == "text_bubble" and getattr(blk, "bubble_xyxy", None) is not None:
        final_padding = 2
        tx1, ty1, tx2, ty2 = [int(round(float(v))) for v in blk.xyxy[:4]]
        ex1 = max(0, tx1 - cx1 - final_padding)
        ey1 = max(0, ty1 - cy1 - final_padding)
        ex2 = min(dilated_crop_mask.shape[1], tx2 - cx1 + final_padding)
        ey2 = min(dilated_crop_mask.shape[0], ty2 - cy1 + final_padding)
        text_envelope = np.zeros(dilated_crop_mask.shape, dtype=bool)
        if ex2 > ex1 and ey2 > ey1:
            text_envelope[ey1:ey2, ex1:ex2] = True

        component_halo = imk.dilate(
            (crop_mask > 0).astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        ) > 0
        text_envelope |= component_halo
        dilated_crop_mask = np.where(text_envelope, dilated_crop_mask, 0).astype(np.uint8)
    return dilated_crop_mask, (cx1, cy1, cx2, cy2)


def collect_block_mask_data(
    img: np.ndarray,
    blk_list: list,
    default_padding: int = 5,
    require_text_or_translation: bool = True,
    clip_to_bubble: bool = True,
) -> list[dict]:
    entries: list[dict] = []
    for blk in blk_list:
        crop_mask, bounds = build_block_mask_data(
            img,
            blk,
            default_padding=default_padding,
            require_text_or_translation=require_text_or_translation,
            clip_to_bubble=clip_to_bubble,
        )
        if crop_mask is None or bounds is None:
            continue
        entries.append({"block": blk, "mask": crop_mask, "bounds": bounds})
    return entries


def generate_mask(img: np.ndarray, blk_list: list, default_padding: int = 5) -> np.ndarray:
    """
    Generate a text-removal mask from filtered connected components and
    only lightly expand it to catch antialiasing around glyph edges.
    Copied verbatim from comic-translate/modules/utils/image_utils.py:538
    """
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for entry in collect_block_mask_data(img, blk_list, default_padding=default_padding, require_text_or_translation=False):
        cx1, cy1, cx2, cy2 = entry["bounds"]
        crop_mask = entry["mask"]
        mask[cy1:cy2, cx1:cx2] = np.bitwise_or(mask[cy1:cy2, cx1:cx2], crop_mask)

    return mask


# ==========================================================================
# COPY-PASTE from comic-translate/pipeline/inpainting.py
# Fast Fill & Hybrid Inpainting Routing
# ==========================================================================

def _get_fast_fill_bounds(block, image: np.ndarray) -> tuple[int, int, int, int] | None:
    base_bounds = getattr(block, "xyxy", None)
    if base_bounds is None or len(base_bounds) < 4:
        return None

    if getattr(block, "text_class", None) == "text_bubble":
        bubble_bounds = getattr(block, "bubble_xyxy", None)
        if bubble_bounds is not None and len(bubble_bounds) >= 4:
            return adjust_text_line_coordinates(bubble_bounds, 10, 10, image)

    return adjust_text_line_coordinates(base_bounds, 10, 10, image)


def _get_fast_fill_regions(
    block,
    bounds: tuple[int, int, int, int],
    crop_mask: np.ndarray,
    residual_crop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if getattr(block, "text_class", None) != "text_bubble":
        return None, None
    if crop_mask is None or residual_crop is None:
        return None, None

    masked_region = (crop_mask > 0) & (residual_crop > 0)
    if not np.any(masked_region):
        return None, None

    h, w = crop_mask.shape[:2]
    x1, y1, _, _ = bounds
    background_region = crop_mask == 0

    text_bounds = getattr(block, "xyxy", None)
    if text_bounds is not None:
        tx1, ty1, tx2, ty2 = [int(round(float(v))) for v in text_bounds]
        lx1 = max(0, min(w, tx1 - x1))
        ly1 = max(0, min(h, ty1 - y1))
        lx2 = max(lx1, min(w, tx2 - x1))
        ly2 = max(ly1, min(h, ty2 - y1))
        block_region = np.zeros((h, w), dtype=bool)
        if lx2 > lx1 and ly2 > ly1:
            block_region[ly1:ly2, lx1:lx2] = True
            masked_in_text = masked_region & block_region
            masked_pixels = int(np.count_nonzero(masked_region))
            masked_in_text_pixels = int(np.count_nonzero(masked_in_text))
            overlap_ratio = (
                masked_in_text_pixels / float(masked_pixels)
                if masked_pixels > 0 else 0.0
            )
            if masked_in_text_pixels >= 8 and overlap_ratio >= 0.6:
                block_background = background_region & block_region
                if np.count_nonzero(block_background) >= 32:
                    background_region = block_background

    halo_kernel = np.ones((3, 3), np.uint8)
    halo = imk.dilate((crop_mask > 0).astype(np.uint8), halo_kernel, iterations=1) > 0
    safe_background = background_region & ~halo
    if np.count_nonzero(safe_background) >= 32:
        background_region = safe_background

    if np.count_nonzero(background_region) < 32:
        return None, None
    return masked_region, background_region


def _estimate_fast_fill_color(
    crop: np.ndarray,
    masked_region: np.ndarray,
    background_region: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    if crop.size == 0 or not np.any(masked_region) or not np.any(background_region):
        return None, "empty-crop-or-regions"

    background_pixels = crop[background_region]
    if background_pixels.shape[0] < 64:
        return None, f"background-too-small:{background_pixels.shape[0]}"

    masked_ratio = float(np.count_nonzero(masked_region)) / float(np.count_nonzero(masked_region | background_region))
    if masked_ratio > 0.90:
        return None, f"masked-ratio-too-high:{masked_ratio:.3f}"

    pixels = background_pixels.reshape(-1, 3)
    pixels_u16 = pixels.astype(np.uint16)
    quantized = pixels_u16 // 16
    keys = (quantized[:, 0] << 8) | (quantized[:, 1] << 4) | quantized[:, 2]
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if unique_keys.size == 0:
        return None, "no-quantized-clusters"

    min_cluster = max(48, int(pixels.shape[0] * 0.015))
    order = np.argsort(counts)[::-1][:16]
    candidates = []
    for key_index in order:
        count = int(counts[key_index])
        if count < min_cluster:
            continue
        cluster = pixels[inverse == key_index].astype(np.float32)
        if cluster.size == 0:
            continue
        median = np.median(cluster, axis=0)
        p10 = np.percentile(cluster, 10, axis=0)
        p90 = np.percentile(cluster, 90, axis=0)
        spread = p90 - p10
        candidates.append({
            "count": count,
            "coverage": count / float(pixels.shape[0]),
            "median": median,
            "brightness": float(np.mean(median)),
            "spread": float(np.max(spread)),
        })

    if not candidates:
        return None, f"no-candidates:min_cluster={min_cluster}"

    candidates.sort(key=lambda item: item["count"], reverse=True)
    largest = candidates[0]
    bright_candidates = [
        item for item in candidates
        if item["brightness"] >= 150.0 and item["count"] >= max(min_cluster, int(largest["count"] * 0.12))
    ]
    selected = max(bright_candidates, key=lambda item: item["count"]) if bright_candidates else largest

    selected_brightness = selected["brightness"]
    if selected_brightness >= 80.0:
        pixel_brightness = 0.299 * pixels[:, 2] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 0]
        valid_pixels = pixels[pixel_brightness >= 50.0]
    else:
        valid_pixels = pixels

    if valid_pixels.size == 0:
        valid_pixels = pixels

    diff = np.abs(valid_pixels - selected["median"])
    is_close = np.all(diff <= 15, axis=1)
    neighborhood_coverage = np.count_nonzero(is_close) / float(valid_pixels.shape[0])

    if neighborhood_coverage < 0.55:
        return None, f"selected-too-low-coverage:coverage={neighborhood_coverage:.3f}"
    if selected["count"] < 96 and selected["coverage"] < 0.03:
        return None, f"selected-too-small:count={selected['count']},coverage={selected['coverage']:.3f}"
    if selected["spread"] > 48.0:
        return None, f"selected-too-wide:spread={selected['spread']:.3f}"

    return selected["median"], (
        "ok:"
        f"count={selected['count']},coverage={selected['coverage']:.3f},"
        f"brightness={selected['brightness']:.1f},spread={selected['spread']:.1f}"
    )


def _is_fast_fill_bubble_background_uniform(
    image: np.ndarray,
    block,
    bounds: tuple[int, int, int, int],
    crop_mask: np.ndarray,
) -> tuple[bool, str]:
    x1, y1, x2, y2 = bounds
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.ndim != 3 or crop.shape[2] < 3:
        return False, "bubble-background-invalid"

    bubble_mask = build_bubble_clip_mask(
        crop.shape[:2],
        bounds,
        block.bubble_xyxy,
        inset=FAST_FILL_BUBBLE_INSET,
        image=image,
        seed_bbox=block.xyxy,
    )
    if bubble_mask is None:
        return False, "bubble-background-unavailable"

    binary_mask = (crop_mask > 0).astype(np.uint8)
    mask_halo = imk.dilate(
        binary_mask,
        np.ones((5, 5), np.uint8),
        iterations=2,
    ) > 0
    sample_region = bubble_mask & ~mask_halo
    sample_count = int(np.count_nonzero(sample_region))
    if sample_count < 64:
        return False, f"bubble-background-too-small:{sample_count}"

    pixels = crop[sample_region, :3].astype(np.float32)
    median = np.median(pixels, axis=0)
    close_to_median = np.all(
        np.abs(pixels - median) <= FAST_FILL_UNIFORM_COLOR_TOLERANCE,
        axis=1,
    )
    uniform_coverage = float(np.count_nonzero(close_to_median)) / float(sample_count)
    if uniform_coverage < FAST_FILL_MIN_UNIFORM_COVERAGE:
        return False, f"bubble-background-nonuniform:coverage={uniform_coverage:.3f}"

    bubble_ys, bubble_xs = np.nonzero(bubble_mask)
    if bubble_ys.size and bubble_xs.size:
        bubble_y1, bubble_y2 = int(bubble_ys.min()), int(bubble_ys.max()) + 1
        bubble_x1, bubble_x2 = int(bubble_xs.min()), int(bubble_xs.max()) + 1
        spatial_min_count = max(64, int(round(sample_count * 0.015)))
        for grid_y in range(3):
            cell_y1 = bubble_y1 + round(grid_y * (bubble_y2 - bubble_y1) / 3)
            cell_y2 = bubble_y1 + round((grid_y + 1) * (bubble_y2 - bubble_y1) / 3)
            for grid_x in range(3):
                cell_x1 = bubble_x1 + round(grid_x * (bubble_x2 - bubble_x1) / 3)
                cell_x2 = bubble_x1 + round((grid_x + 1) * (bubble_x2 - bubble_x1) / 3)
                cell_region = sample_region[cell_y1:cell_y2, cell_x1:cell_x2]
                cell_count = int(np.count_nonzero(cell_region))
                if cell_count < spatial_min_count:
                    continue
                cell_pixels = crop[cell_y1:cell_y2, cell_x1:cell_x2][cell_region, :3].astype(np.float32)
                cell_median = np.median(cell_pixels, axis=0)
                median_shift = float(np.max(np.abs(cell_median - median)))
                if median_shift > FAST_FILL_UNIFORM_COLOR_TOLERANCE:
                    return False, f"bubble-background-nonuniform:spatial-color-shift={median_shift:.1f}"

    masked_bubble = bubble_mask & (binary_mask > 0)
    bubble_count = int(np.count_nonzero(bubble_mask))
    masked_count = int(np.count_nonzero(masked_bubble))
    masked_coverage = masked_count / float(max(1, bubble_count))
    if masked_count >= 64 and masked_coverage >= 0.20:
        masked_median = np.median(crop[masked_bubble, :3].astype(np.float32), axis=0)
        median_shift = float(np.max(np.abs(masked_median - median)))
        if median_shift > FAST_FILL_UNIFORM_COLOR_TOLERANCE:
            return False, f"bubble-background-nonuniform:expanded-mask-color-shift={median_shift:.1f}"

    return True, f"bubble-background-uniform:coverage={uniform_coverage:.3f}"


def _get_fast_fill_uniformity_mask(
    image: np.ndarray,
    block,
    bounds: tuple[int, int, int, int],
    fallback_mask: np.ndarray,
) -> np.ndarray:
    try:
        block_mask, block_bounds = build_block_mask_data(
            image,
            block,
            require_text_or_translation=False,
            clip_to_bubble=True,
        )
    except Exception as exc:
        logger.debug("Inpaint fast-fill: failed to rebuild routing mask: %s", exc)
        return fallback_mask

    if block_mask is None or block_bounds is None or not np.any(block_mask):
        return fallback_mask

    x1, y1, x2, y2 = [int(v) for v in bounds]
    bx1, by1, bx2, by2 = [int(v) for v in block_bounds]
    ix1, iy1 = max(x1, bx1), max(y1, by1)
    ix2, iy2 = min(x2, bx2), min(y2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return fallback_mask

    routing_mask = np.zeros_like(fallback_mask)
    routing_mask[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = block_mask[
        iy1 - by1:iy2 - by1,
        ix1 - bx1:ix2 - bx1,
    ]
    return routing_mask if np.any(routing_mask) else fallback_mask


def _get_associated_residual_components(residual_crop: np.ndarray, masked_region: np.ndarray) -> np.ndarray:
    residual_binary = (residual_crop > 0).astype(np.uint8)
    if not np.any(residual_binary):
        return masked_region

    num_labels, labels, _stats, _centroids = imk.connected_components_with_stats(
        residual_binary, connectivity=8
    )
    if num_labels <= 1:
        return residual_binary > 0

    near_mask = imk.dilate(
        masked_region.astype(np.uint8),
        np.ones((17, 17), np.uint8),
        iterations=1,
    ) > 0
    overlap = near_mask & (labels > 0)
    overlap_labels = np.unique(labels[overlap])
    if overlap_labels.size == 0:
        return masked_region

    return np.isin(labels, overlap_labels)


def _fast_fill_block(
    cleaned_image: np.ndarray,
    residual_mask: np.ndarray,
    block,
    bounds: tuple[int, int, int, int],
    crop_mask: np.ndarray,
) -> tuple[bool, str]:
    x1, y1, x2, y2 = bounds
    crop = cleaned_image[y1:y2, x1:x2]
    residual_crop = residual_mask[y1:y2, x1:x2]
    masked_region, background_region = _get_fast_fill_regions(block, bounds, crop_mask, residual_crop)
    if masked_region is None or background_region is None:
        return False, "invalid-regions"

    fill_color, color_reason = _estimate_fast_fill_color(crop, masked_region, background_region)
    if fill_color is None:
        return False, color_reason

    fill_region = _get_associated_residual_components(residual_crop, masked_region)
    applied_region = fill_region

    if getattr(block, "text_class", None) == "text_bubble" and getattr(block, "bubble_xyxy", None) is not None:
        bubble_mask = build_bubble_clip_mask(
            fill_region.shape[:2],
            bounds,
            block.bubble_xyxy,
            inset=FAST_FILL_BUBBLE_INSET,
            image=cleaned_image,
            seed_bbox=block.xyxy,
        )
        if bubble_mask is not None:
            num_labels, labeled_fill = imk.connected_components(fill_region, connectivity=4)
            overlapping_labels = np.unique(labeled_fill[bubble_mask])
            keep_labels = overlapping_labels[overlapping_labels > 0]
            if keep_labels.size > 0:
                applied_region = np.isin(labeled_fill, keep_labels) & bubble_mask
            else:
                applied_region = np.zeros_like(fill_region)
    else:
        bubble_mask = None

    soft_mask = imk.gaussian_blur(applied_region.astype(np.uint8) * 255, 1.0).astype(np.float32) / 255.0
    soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., np.newaxis]

    if bubble_mask is not None:
        soft_mask = soft_mask * bubble_mask[..., np.newaxis]
    crop_f = crop.astype(np.float32)
    fill_rgb = np.broadcast_to(fill_color, crop.shape).astype(np.float32)
    blended = crop_f * (1.0 - soft_mask) + fill_rgb * soft_mask
    cleaned_image[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    residual_crop[applied_region] = 0
    return True, color_reason


def _apply_fast_bubble_cleanup(
    image: np.ndarray,
    mask: np.ndarray,
    blk_list: list | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if image is None or mask is None or not np.any(mask) or not blk_list:
        return image.copy(), mask.copy(), 0

    cleaned_image = image.copy()
    residual_mask = mask.copy()
    cleaned_blocks = 0
    cleaned_bubble_blocks = []

    for idx, block in enumerate(blk_list):
        if getattr(block, "xyxy", None) is None or len(block.xyxy) < 4:
            continue
        if getattr(block, "text_class", None) != "text_bubble" or getattr(block, "bubble_xyxy", None) is None:
            continue
        bounds = _get_fast_fill_bounds(block, image)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        residual_crop = residual_mask[y1:y2, x1:x2]
        if not np.any(residual_crop):
            continue
        crop_mask = np.where(residual_crop > 0, 255, 0).astype(np.uint8)
        background_is_uniform, uniformity_reason = _is_fast_fill_bubble_background_uniform(
            image, block, bounds, crop_mask
        )
        if background_is_uniform:
            uniformity_mask = _get_fast_fill_uniformity_mask(image, block, bounds, crop_mask)
            background_is_uniform, uniformity_reason = _is_fast_fill_bubble_background_uniform(
                image, block, bounds, uniformity_mask
            )
        if not background_is_uniform:
            logger.info("Inpaint fast-fill: block[%d] routed to NN (%s)", idx, uniformity_reason)
            continue

        if getattr(block, "text_class", None) == "text_bubble" and getattr(block, "bubble_xyxy", None) is not None:
            crop_mask = clip_mask_components_to_bubble(
                crop_mask,
                bounds,
                block.bubble_xyxy,
                inset=FAST_FILL_BUBBLE_INSET,
                image=image,
                seed_bbox=block.xyxy,
            )

        initial_overlap = int(np.count_nonzero(crop_mask))
        if initial_overlap <= 0:
            continue
        success, reason = _fast_fill_block(cleaned_image, residual_mask, block, bounds, crop_mask)
        if not success:
            fallback_mask, fallback_bounds = build_block_mask_data(
                image, block, require_text_or_translation=False, clip_to_bubble=True
            )
            if fallback_mask is None or fallback_bounds is None:
                continue
            success, fallback_reason = _fast_fill_block(
                cleaned_image, residual_mask, block, fallback_bounds, fallback_mask
            )
            if not success:
                continue
            reason = f"fallback:{fallback_reason}"

        cleaned_blocks += 1
        cleaned_bubble_blocks.append(block)

    if cleaned_blocks:
        bubble_scope = np.zeros(mask.shape, dtype=bool)
        bubble_allowed = np.zeros(mask.shape, dtype=bool)
        for bubble_block in cleaned_bubble_blocks:
            if getattr(bubble_block, "text_class", None) != "text_bubble" or getattr(bubble_block, "bubble_xyxy", None) is None:
                continue
            bubble_bounds = _get_fast_fill_bounds(bubble_block, image)
            if bubble_bounds is None:
                continue
            bx1, by1, bx2, by2 = bubble_bounds
            bubble_scope[by1:by2, bx1:bx2] = True
            bubble_clip = build_bubble_clip_mask((by2 - by1, bx2 - bx1), bubble_bounds, bubble_block.bubble_xyxy, inset=FAST_FILL_BUBBLE_INSET, image=image, seed_bbox=bubble_block.xyxy)
            if bubble_clip is not None:
                bubble_allowed[by1:by2, bx1:bx2] |= bubble_clip
        spill = (residual_mask > 0) & bubble_scope & ~bubble_allowed
        if np.any(spill):
            residual_mask[spill] = 0
    return cleaned_image, residual_mask, cleaned_blocks


# ==========================================================================
# NN LaMa Inpainting Execution (Full-image & Patch-based)
# ==========================================================================

def _run_region_sync(
    inpainter_key: str,
    crop_img: np.ndarray,
    crop_mask: np.ndarray,
    config: Config,
    crop_size: int,
    device: str,
) -> np.ndarray:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            dispatch_inpainting(
                inpainter_key, crop_img, crop_mask, config, crop_size, device, verbose=False
            )
        )
    finally:
        loop.close()


def apply_solid_fill(img_rgb: np.ndarray, pts_solid: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Erases text by filling a polygon with a solid color (legacy utility)."""
    fill_color = np.round(color).astype(np.uint8).tolist()
    cv2.fillPoly(img_rgb, [pts_solid], fill_color)
    return img_rgb


async def inpaint_image(
    image: np.ndarray,
    mask: np.ndarray,
    blk_list: list | None = None,
) -> np.ndarray:
    """
    Comic-translate exact inpaint_image pipeline:
    1. _apply_fast_bubble_cleanup() -> solid-fill uniform bubbles
    2. Residual mask -> patch-based or full-image NN LaMa inpainting
    """
    if image is None:
        return None
    if mask is None or not np.any(mask):
        return image.copy()

    # Step 1: Fast Fill
    working_image, working_mask, cleaned_blocks = _apply_fast_bubble_cleanup(image, mask, blk_list)
    if cleaned_blocks:
        logger.info(f"[Inpaint Hybrid] Fast-cleaned {cleaned_blocks} uniform bubble block(s).")
    if working_mask is None or not np.any(working_mask):
        return working_image

    contours, _ = imk.find_contours(working_mask)
    if not contours:
        return working_image

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    config = Config()
    inpainter_key = config.inpainter.inpainter

    # Pre-load LaMa model
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.load(device)

    sem = _get_lama_semaphore()
    async with sem:
        boxes = [imk.bounding_rect(c) for c in contours]
        merged_boxes = merge_overlapping_padded_boxes(boxes, working_image.shape)

        h_img, w_img = working_image.shape[:2]
        max_size = 1024
        scale = min(1.0, max_size / max(h_img, w_img))
        full_pixels = max(1, int(round(w_img * scale))) * max(1, int(round(h_img * scale)))

        total_patch_pixels = 0
        for x1, y1, x2, y2 in merged_boxes:
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            eff_w = max(128, int(np.ceil(w / 8.0) * 8))
            eff_h = max(128, int(np.ceil(h / 8.0) * 8))
            total_patch_pixels += eff_w * eff_h

        session_overhead_penalty = (len(merged_boxes) - 1) * 50000
        estimated_patch_cost = total_patch_pixels + session_overhead_penalty
        use_patches = estimated_patch_cost < full_pixels

        inpainted = working_image.copy()

        if use_patches:
            logger.info(f"[Inpaint Hybrid] Running patch-based LaMa ({len(merged_boxes)} patch(es)).")
            for x1, y1, x2, y2 in merged_boxes:
                img_patch = working_image[y1:y2, x1:x2]
                mask_patch = working_mask[y1:y2, x1:x2]
                patch_size = max(img_patch.shape[:2])

                patch_inpainted = await asyncio.to_thread(
                    _run_region_sync, inpainter_key, img_patch, mask_patch, config, patch_size, device
                )
                inpainted[y1:y2, x1:x2] = patch_inpainted
        else:
            logger.info("[Inpaint Hybrid] Running full-image LaMa.")
            full_size = max(h_img, w_img)
            inpainted = await asyncio.to_thread(
                _run_region_sync, inpainter_key, working_image, working_mask, config, full_size, device
            )

    return inpainted


# Backwards compatibility alias
async def run_local_lama_inpainting(img_rgb: np.ndarray, full_regions_pts: list[np.ndarray]) -> np.ndarray:
    """Fallback legacy wrapper if called directly."""
    if not full_regions_pts:
        return img_rgb
    h, w, _ = img_rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in full_regions_pts:
        cv2.fillPoly(mask, [pts], 255)
    return await inpaint_image(img_rgb, mask, blk_list=None)
