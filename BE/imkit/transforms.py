"""Image transformation operations for the imkit module."""

from __future__ import annotations
import cv2
import numpy as np
# import mahotas as mh
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional, Sequence, Union
from .utils import ensure_uint8


def to_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale conversion using Pillow."""
    if img.ndim == 3:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        pil_img = Image.fromarray(img)
        gray = pil_img.convert("L")  # Pillow grayscale
        return np.array(gray, dtype=np.uint8)
    elif img.dtype != np.uint8:
        return img.astype(np.uint8)
    return img.copy()


def gaussian_blur(array: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to an image array."""
    im = Image.fromarray(ensure_uint8(array))
    return np.array(im.filter(ImageFilter.GaussianBlur(radius=radius)))


def resize(
    image: np.ndarray, 
    size: tuple[int, int], 
    mode: Image.Resampling = Image.Resampling.LANCZOS
) -> np.ndarray:
    """Resize an image array to the specified size."""
    w, h = size
    im = Image.fromarray(ensure_uint8(image))
    im = im.resize((w, h), resample=mode)
    return np.array(im)


def lut(array: np.ndarray, lookup_table: np.ndarray) -> np.ndarray:
    """
    Apply lookup table transformation.
    Replaces cv2.LUT functionality.
    
    Args:
        array: Input array
        lookup_table: Lookup table for transformation
        
    Returns:
        Transformed array
    """
    return lookup_table[array]


def merge_channels(channels: list) -> np.ndarray:
    """
    Merge separate channels into a multi-channel image.
    Replaces cv2.merge functionality.
    
    Args:
        channels: List of single-channel arrays
        
    Returns:
        Multi-channel array
    """
    return np.stack(channels, axis=-1)


def _monotone_chain(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain convex hull. 
    Input Nx2 array, returns hull vertices CCW (no duplicate last point).
    """
    pts = np.asarray(points, dtype=np.float64)
    # Handle OpenCV-style contour format (N, 1, 2) -> (N, 2)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts[:, 0, :]
    if pts.shape[0] <= 1:
        return pts.copy()
    # sort lexicographically by x then y
    pts_sorted = np.array(sorted(map(tuple, pts)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def min_area_rect(points, assume_hull=False):
    """
    Compute minimum-area bounding rectangle for a set of 2D points.

    Notes:
        - Uses a rotating-calipers style sweep over edges of the convex hull to find
            the rectangle of minimal area.
        - Tries to match OpenCV's cv2.minAreaRect conventions for (width, height, angle),
            including some quirks for degenerate 1- and 2-point inputs.

    Returns:
        rect = ((cx, cy), (w, h), angle) with same convention as cv2.minAreaRect:
                - angle in [0, 90) degrees  
                - width/height correspond to the edges, no forced ordering
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("no points")
    if not assume_hull:
        hull = _monotone_chain(pts)
    else:
        hull = pts.copy()

    m = hull.shape[0]
    if m == 0:
        raise ValueError("empty hull")
    if m == 1:
        x, y = hull[0]
        rect = ((x, y), (0.0, 0.0), 0.0)
        return rect
    if m == 2:
        (x0, y0), (x1, y1) = hull
        dx, dy = x1 - x0, y1 - y0
        
        # For degenerate case, match cv2's exact behavior
        edge_len = np.hypot(dx, dy)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        
        # cv2 puts the line length in width, zero in height for degenerate cases
        width = edge_len
        height = 0.0
        
        # cv2 uses the line angle directly for degenerate cases (not perpendicular)
        angle = np.degrees(np.arctan2(dy, dx))
        
        # cv2 applies specific angle transformations for degenerate cases
        if abs(dx) < 1e-10:  # vertical line
            angle = -90.0 if dy > 0 else 90.0
        elif abs(dy) < 1e-10:  # horizontal line  
            angle = 180.0 if dx > 0 else 0.0  # cv2 uses 180° for positive horizontal lines
        else:  # diagonal line - cv2 shifts by 180° 
            angle -= 180.0  # cv2 consistently subtracts 180° from arctan2 result

        rect = ((cx, cy), (width, height), angle)
        return rect

    # edges
    # Consider each edge direction from the hull as a candidate rectangle orientation
    idx_next = (np.arange(m) + 1) % m
    edges = hull[idx_next] - hull[np.arange(m)]
    edge_len = np.hypot(edges[:, 0], edges[:, 1])
    valid = edge_len > 1e-12
    edges = edges[valid]
    edge_len = edge_len[valid]
    if edges.shape[0] == 0:
        x, y = hull[0]
        rect = ((x, y), (0.0, 0.0), 0.0)
        return rect

    # Unit vectors for candidate x-axes (ux) and corresponding y-axes (uy)
    ux = edges / edge_len[:, None]
    uy = np.column_stack((-ux[:, 1], ux[:, 0]))

    # Project hull onto candidate axes and compute bounding intervals
    proj_x = hull.dot(ux.T)
    proj_y = hull.dot(uy.T)

    min_x = proj_x.min(axis=0)
    max_x = proj_x.max(axis=0)
    min_y = proj_y.min(axis=0)
    max_y = proj_y.max(axis=0)

    widths = max_x - min_x
    heights = max_y - min_y
    areas = widths * heights

    k = int(np.argmin(areas))
    best_ux = ux[k]
    best_uy = uy[k]

    cx_rot = 0.5 * (min_x[k] + max_x[k])
    cy_rot = 0.5 * (min_y[k] + max_y[k])
    center = np.dot([cx_rot, cy_rot], np.column_stack((best_ux, best_uy)).T)

    # Get the dimensions along each axis
    dim_along_ux = float(widths[k])   # dimension along best_ux
    dim_along_uy = float(heights[k])  # dimension along best_uy (perpendicular)
    
    # Calculate angle of best_ux from horizontal
    angle_ux = float(np.degrees(np.arctan2(best_ux[1], best_ux[0])))
    
    # Normalize angle_ux to range [-180, 180)
    while angle_ux < -180:
        angle_ux += 360
    while angle_ux >= 180:
        angle_ux -= 360
    
    # OpenCV's cv2.minAreaRect convention:
    # For axis-aligned rectangles, cv2 uses angle=90.0 and swaps the dimensions
    # (width gets the y-extent, height gets the x-extent)
    # For rotated rectangles, angle indicates the rotation of the first (width) edge
    
    # Check if this is axis-aligned (angle near 0 or 90 degrees)
    is_horizontal = abs(angle_ux) < 1e-6 or abs(abs(angle_ux) - 180) < 1e-6
    is_vertical = abs(abs(angle_ux) - 90) < 1e-6
    
    if is_horizontal:
        # Horizontal edge: cv2 uses angle=90.0 and swaps dimensions
        # width = vertical extent, height = horizontal extent
        width = dim_along_uy
        height = dim_along_ux
        angle = 90.0
    elif is_vertical:
        # Vertical edge: cv2 uses angle=90.0 with standard dimensions
        # width = horizontal extent, height = vertical extent  
        width = dim_along_ux
        height = dim_along_uy
        angle = 90.0
    else:
        # Non-axis-aligned: convert angle to range (0, 90]
        width = dim_along_ux
        height = dim_along_uy
        angle = angle_ux
        
        # If angle is negative, add 90 and swap dimensions
        if angle < 0:
            angle += 90.0
            width, height = height, width

    rect = (tuple(center), (width, height), angle)

    return rect


def box_points(rect: tuple) -> np.ndarray:
    """
    Get corner points of a rotated rectangle.
    This is a pure numpy implementation that replaces cv2.boxPoints.
    
    The `rect` input is expected to be in the format returned by cv2.minAreaRect:
    ((center_x, center_y), (width, height), angle_in_degrees)
    
    Args:
        rect: A tuple containing the center, size, and angle of the rectangle.
        
    Returns:
        A NumPy array of shape (4, 2) with the 4 corner points.
    """
    # Unpack the rectangle data
    (center_x, center_y), (width, height), angle = rect
    center = np.array([center_x, center_y])
    
    # Convert the angle to radians
    # Note: cv2.minAreaRect returns angle in degrees in range [-90, 0)
    theta = np.deg2rad(angle)
    
    c, s = np.cos(theta), np.sin(theta)
    
    # Create the rotation matrix
    # This matrix is used to rotate points around the origin
    rotation_matrix = np.array([[c, -s], 
                                [s, c]])
    
    # Define the half-width and half-height
    half_w, half_h = width / 2, height / 2
    
    # Define the 4 corners of the box in its local, unrotated coordinate system (centered at origin)
    unrotated_points = np.array([
        [-half_w, -half_h], # Bottom-left
        [ half_w, -half_h], # Bottom-right
        [ half_w,  half_h], # Top-right
        [-half_w,  half_h]  # Top-left
    ])
    
    # Rotate the points around the origin
    # We use matrix multiplication (the @ operator)
    # The result is a 4x2 matrix of rotated points
    rotated_points = unrotated_points @ rotation_matrix.T
    
    # Translate the points to the rectangle's center
    box = rotated_points + center
    
    return box.astype(np.float32)


def fill_poly(image: np.ndarray, pts: list | np.ndarray, color: int = 1) -> np.ndarray:
    if isinstance(pts, np.ndarray):
        polygons = [pts.astype(np.int32)]
    else:
        polygons = [np.ascontiguousarray(p, dtype=np.int32) for p in pts]
    return cv2.fillPoly(image, polygons, color)


def connected_components(image: np.ndarray, connectivity: int = 4) -> tuple:
    img_u8 = (image > 0).astype(np.uint8) if image.dtype != np.uint8 else image
    num_labels, labels = cv2.connectedComponents(img_u8, connectivity=connectivity)
    return num_labels, labels


def connected_components_with_stats(image: np.ndarray, connectivity: int = 4) -> tuple:
    img_u8 = (image > 0).astype(np.uint8) if image.dtype != np.uint8 else image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_u8, connectivity=connectivity)
    return num_labels, labels, stats, centroids


def line(
    image: np.ndarray, 
    pt1: tuple, 
    pt2: tuple, 
    color: int, 
    thickness: int = 1
) -> np.ndarray:
    """
    Draw a line on an image using PIL.
    Replaces cv2.line functionality.
    
    Args:
        image: Target image
        pt1: First point (x, y)
        pt2: Second point (x, y)
        color: Line color
        thickness: Line thickness
        
    Returns:
        Image with line drawn
    """
    
    pil_image = Image.fromarray(ensure_uint8(image))
    draw = ImageDraw.Draw(pil_image)
    draw.line([pt1, pt2], fill=color, width=thickness)

    return np.array(pil_image)


def convert_scale_abs(
    array: np.ndarray, 
    alpha: float = 1.0, 
    beta: float = 0.0
) -> np.ndarray:
    """
    Convert array to absolute values with scaling.
    Replaces cv2.convertScaleAbs functionality.
    
    Args:
        array: Input array
        alpha: Scale factor (default 1.0)
        beta: Offset value (default 0.0)
        
    Returns:
        Scaled and converted array as uint8
    """
    # Apply scaling and offset
    scaled = array * alpha + beta
    
    # Convert to absolute values and clip to uint8 range
    abs_scaled = np.abs(scaled)
    clipped = np.clip(abs_scaled, 0, 255)
    
    return clipped.astype(np.uint8)


def threshold(
    array: np.ndarray, 
    thresh: float, 
    maxval: float = 255, 
    thresh_type: int = 0
) -> tuple[float, np.ndarray]:
    """
    Apply threshold to an array.
    Replaces cv2.threshold functionality.
    
    Args:
        array: Input array
        thresh: Threshold value
        maxval: Maximum value to use with thresholding type
        thresh_type: Thresholding type (0 = binary)
        
    Returns:
        Tuple of (threshold_value, thresholded_array)
    """
    if array.ndim == 3:
        array = to_gray(array)
    
    # Binary threshold (thresh_type = 0)
    result = np.where(array > thresh, maxval, 0).astype(np.uint8)
    
    return thresh, result


def otsu_threshold(array: np.ndarray) -> tuple[float, np.ndarray]:
    if array.ndim == 3:
        array = to_gray(array)
    thresh_val, result = cv2.threshold(array.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thresh_val), result


def rectangle(
    image: np.ndarray, 
    pt1: tuple, 
    pt2: tuple, 
    color: Optional[tuple|int], 
    thickness: int = 1
) -> np.ndarray:
    """
    Mimics cv2.rectangle() using PIL.ImageDraw.Draw.rectangle().

    Args:
        image (np.ndarray): The input image as a numpy array.
        pt1 (tuple): The top-left corner coordinates (x, y).
        pt2 (tuple): The bottom-right corner coordinates (x, y).
        color (tuple): The rectangle color in BGR format (e.g., (255, 0, 0) for blue).
        thickness (int, optional): The thickness of the line. 
                                  If a negative number (e.g., -1), the rectangle is filled.
                                  Defaults to 1.

    Returns:
        np.ndarray: The modified image as a numpy array.
    """
    # Create an ImageDraw object
    img_pil = Image.fromarray(ensure_uint8(image))
    draw = ImageDraw.Draw(img_pil)
    
    # Normalize color to what PIL expects depending on image mode.
    # Acceptable inputs:
    #  - int (grayscale or single-value for RGB)
    #  - tuple/list of length 1 (grayscale) or 3 (BGR order expected, will be converted to RGB)
    if color is None:
        color = 1

    mode = img_pil.mode  # e.g. 'L', 'RGB', 'RGBA'

    # Normalize numeric and sequence types
    if isinstance(color, int):
        if mode in ("RGB", "RGBA"):
            pil_color = (int(color),) * 3
        else:
            pil_color = int(color)
    elif isinstance(color, (tuple, list, np.ndarray)):
        col = tuple(int(x) for x in color)
        if len(col) == 3:
            # assume input is BGR (OpenCV-style) -> convert to RGB for PIL
            pil_color = (col[2], col[1], col[0])
        elif len(col) == 1:
            if mode in ("RGB", "RGBA"):
                v = col[0]
                pil_color = (v, v, v)
            else:
                pil_color = col[0]
        else:
            raise ValueError("Color tuple must have length 1 or 3 for grayscale or RGB images.")
    else:
        raise ValueError("Color must be an int or a tuple/list/ndarray of length 1 or 3.")

    if thickness == -1:
        # Draw a filled rectangle
        draw.rectangle([pt1, pt2], fill=pil_color)
    elif thickness > 0:
        # Draw an outlined rectangle with a specified width
        draw.rectangle([pt1, pt2], outline=pil_color, width=thickness)

    return np.array(img_pil)


def add_weighted(
    src1: np.ndarray, 
    alpha: float, 
    src2: np.ndarray, 
    beta: float, 
    gamma: float
) -> np.ndarray:
    """
    Implements cv2.addWeighted() using NumPy.

    Args:
        src1 (np.ndarray): First input array.
        alpha (float): Weight for the first array elements.
        src2 (np.ndarray): Second input array.
        beta (float): Weight for the second array elements.
        gamma (float): Scalar added to the weighted sum.

    Returns:
        np.ndarray: The weighted sum of the two arrays, with the same data type
                    as the input arrays, and values clipped to the valid range.
    """
    # Ensure src1 and src2 have the same dimensions and data type.
    if src1.shape != src2.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Perform the weighted sum using NumPy.
    # Arithmetic operations will be performed on floats to prevent overflow
    # before the final saturation.
    weighted_sum = (alpha * src1.astype(np.float64) +
                    beta * src2.astype(np.float64) +
                    gamma)

    # Re-cast to the original data type and clip values to handle saturation.
    # This prevents the modulo arithmetic behavior of standard NumPy integer operations.
    output = np.clip(weighted_sum,
                     np.iinfo(src1.dtype).min,
                     np.iinfo(src1.dtype).max)

    return output.astype(src1.dtype)