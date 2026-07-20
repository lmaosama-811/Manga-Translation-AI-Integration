"""
Module: app.utils.image
Description: Utility functions for image loading, resizing/compression, and Base64 conversions.
"""

import io
import base64
from PIL import Image

def read_image_to_pil(contents: bytes) -> Image.Image:
    """
    Reads raw bytes into a PIL Image and normalizes RGBA/P to RGB format.
    """
    image = Image.open(io.BytesIO(contents))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    return image

def compress_and_encode_image(image: Image.Image, max_size: int = 2048, quality: int = 80) -> str:
    """
    Resizes image within max_size constraint and compresses it into a high-efficiency JPEG Base64 string for VLM payload.
    """
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def encode_pil_to_base64(image: Image.Image, quality: int = 90) -> str:
    """
    Encodes any PIL Image to a Base64 encoded JPEG string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
