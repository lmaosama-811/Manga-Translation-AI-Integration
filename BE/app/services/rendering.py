"""
Module: app.services.rendering
Description: Controls the typography rendering system, including monkeypatching the core engine to fix word stretching.
"""

import numpy as np
from manga_translator.rendering import dispatch as dispatch_rendering
from manga_translator.rendering import text_render
from manga_translator.rendering.text_render import calc_horizontal as original_calc_horizontal

# Custom horizontal calculation to support pre-divided Vietnamese strings with newline characters '\n'
def custom_calc_horizontal(font_size: int, text: str, max_width: int, max_height: int, language: str = 'en_US', hyphenate: bool = True):
    if '\\n' in text or '\n' in text:
        lines = text.replace('\\n', '\n').split('\n')
        widths = [text_render.get_string_width(font_size, line) for line in lines]
        return lines, widths
    return original_calc_horizontal(font_size, text, max_width, max_height, language, hyphenate)

# Run Monkeypatching directly replacing the core calculation in the original project's text rendering engine
text_render.calc_horizontal = custom_calc_horizontal

def init_rendering_engine(font_path: str):
    """
    Sets the primary font path for the underlying text renderer.
    """
    text_render.set_font(font_path)

async def render_text_bubbles(
    img_inpainted: np.ndarray,
    text_regions: list,
    font_path: str,
    line_spacing: float = 0.20
) -> np.ndarray:
    """
    Renders typeset translated text bubbles back onto the inpainted image.
    Uses generous line spacing to emulate high-quality Vietnamese manga translations.
    """
    return await dispatch_rendering(
        img=img_inpainted,
        text_regions=text_regions,
        font_path=font_path,
        font_size_fixed=None,
        font_size_offset=0,
        font_size_minimum=0,
        hyphenate=True,
        line_spacing=line_spacing,
        disable_font_border=False
    )
