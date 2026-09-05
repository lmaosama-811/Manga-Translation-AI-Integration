"""Morphological operations for the imkit module using OpenCV."""

from __future__ import annotations
import cv2
import numpy as np

MORPH_RECT = cv2.MORPH_RECT
MORPH_CROSS = cv2.MORPH_CROSS
MORPH_ELLIPSE = cv2.MORPH_ELLIPSE

MORPH_OPEN = cv2.MORPH_OPEN
MORPH_CLOSE = cv2.MORPH_CLOSE
MORPH_GRADIENT = cv2.MORPH_GRADIENT
MORPH_TOPHAT = cv2.MORPH_TOPHAT
MORPH_BLACKHAT = cv2.MORPH_BLACKHAT


def dilate(mask: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), kernel.astype(np.uint8), iterations=iterations)


def erode(mask: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    return cv2.erode(mask.astype(np.uint8), kernel.astype(np.uint8), iterations=iterations)


def morphology_ex(image: np.ndarray, op: any, kernel: np.ndarray) -> np.ndarray:
    if isinstance(op, str):
        op_map = {
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
            'gradient': cv2.MORPH_GRADIENT,
            'tophat': cv2.MORPH_TOPHAT,
            'blackhat': cv2.MORPH_BLACKHAT,
        }
        op = op_map.get(op, cv2.MORPH_CLOSE)
    return cv2.morphologyEx(image.astype(np.uint8), op, kernel.astype(np.uint8))


def get_structuring_element(shape: int, ksize: tuple) -> np.ndarray:
    return cv2.getStructuringElement(shape, ksize)


def close_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask using cv2 floodFill."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_uint8.shape[:2]
    flood = mask_uint8.copy()
    bg_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, bg_mask, (0, 0), 255)
    inv_flood = cv2.bitwise_not(flood)
    out = cv2.bitwise_or(mask_uint8, inv_flood)
    return out > 0
