"""
test_grid_overlay.py
Chay: python test_grid_overlay.py <duong_dan_anh>
Ket qua: luu anh da co grid overlay ra Desktop voi ten <ten_goc>_grid.png
"""

import sys
import os

# Fix encoding cho Windows terminal
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from PIL import Image

# Them thu muc goc vao sys.path de import duoc app.utils.grid_overlay
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from BE.app.utils.grid_overlay import apply_grid_overlay

def main():
    if len(sys.argv) < 2:
        print("Cach dung: python test_grid_overlay.py <duong_dan_anh>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Loi: khong tim thay file '{input_path}'")
        sys.exit(1)

    print(f"Dang load anh: {input_path}")
    image = Image.open(input_path).convert("RGB")
    print(f"Kich thuoc goc: {image.size[0]}x{image.size[1]} px")

    print("Dang ap dung grid overlay...")
    annotated, _b64 = apply_grid_overlay(image)

    # Luu ra Desktop
    desktop = Path(os.path.expanduser("~")) / "Desktop"
    out_name = input_path.stem + "_grid.png"
    out_path = desktop / out_name

    annotated.save(str(out_path), format="PNG")
    print(f"Done! Da luu: {out_path}")

if __name__ == "__main__":
    main()
