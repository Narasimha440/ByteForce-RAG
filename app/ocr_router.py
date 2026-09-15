"""
Adaptive Document Router for Industrial OCR Pipeline — SIH 26117.

Classifies each page/image into a document type BEFORE running OCR,
then assigns the appropriate DPI and preprocessing tier.

This avoids wasting expensive heavy preprocessing on clean digital PDFs
while ensuring handwritten inspection sheets and field photographs get
the full denoising + deskew treatment.

Classification is 100% local, zero-ML, heuristic-based — no GPU required.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────────────────

class PageType(str, Enum):
    DIGITAL_CLEAN       = "digital_clean"        # Rendered from digital PDF, high-quality
    SCANNED_LIGHT       = "scanned_light"         # Scanned, good quality, minor skew
    SCANNED_HEAVY       = "scanned_heavy"         # Faded, noisy, carbon-copy, old scan
    HANDWRITTEN         = "handwritten"            # Handwritten inspection/field notes
    ENGINEERING_DRAWING = "engineering_drawing"    # P&ID, schematic, large-format A3


class PreprocessTier(str, Enum):
    SKIP   = "skip"    # Pass image directly — no preprocessing needed
    LIGHT  = "light"   # Autocontrast + mild sharpening + deskew if tilted
    HEAVY  = "heavy"   # Denoise + Otsu binarization + deskew + adaptive threshold


# ── Adaptive DPI table ────────────────────────────────────────────────────────
# Key insight: digital PDFs don't need high DPI — they're already vector-precise.
# Engineering drawings need the highest DPI to resolve fine annotation text.

ADAPTIVE_DPI: Dict[PageType, int] = {
    PageType.DIGITAL_CLEAN:       150,   # Fast — vectors, no loss at 150
    PageType.SCANNED_LIGHT:       200,   # Standard scan quality
    PageType.SCANNED_HEAVY:       300,   # Higher res to recover faded detail
    PageType.HANDWRITTEN:         300,   # Higher res for irregular strokes
    PageType.ENGINEERING_DRAWING: 350,   # Highest — fine annotation text, leader lines
}

ADAPTIVE_PREPROCESS_TIER: Dict[PageType, PreprocessTier] = {
    PageType.DIGITAL_CLEAN:       PreprocessTier.SKIP,
    PageType.SCANNED_LIGHT:       PreprocessTier.LIGHT,
    PageType.SCANNED_HEAVY:       PreprocessTier.HEAVY,
    PageType.HANDWRITTEN:         PreprocessTier.HEAVY,
    PageType.ENGINEERING_DRAWING: PreprocessTier.LIGHT,
}


# ── Classification Heuristics ─────────────────────────────────────────────────

def _image_stats(image: Image.Image) -> Dict[str, Any]:
    """
    Compute lightweight pixel statistics for classification.
    Operates on a downsampled thumbnail for speed.
    """
    import numpy as np

    # Downsample to at most 400px wide for fast analysis
    thumb = image.copy()
    thumb.thumbnail((400, 400), Image.LANCZOS)
    arr = np.array(thumb.convert("L"), dtype=np.float32)

    pixel_mean   = float(arr.mean())
    pixel_std    = float(arr.std())
    pixel_min    = float(arr.min())
    pixel_max    = float(arr.max())
    dynamic_range = pixel_max - pixel_min

    # Estimate noise level using Laplacian variance (higher = sharper / noisier)
    try:
        import cv2
        lap_var = float(cv2.Laplacian(arr.astype("uint8"), cv2.CV_64F).var())
    except ImportError:
        lap_var = pixel_std  # fallback

    # Estimate text darkness ratio: fraction of pixels < 80 (dark ink on white)
    dark_pixel_ratio = float((arr < 80).sum() / arr.size)

    return {
        "mean":           pixel_mean,
        "std":            pixel_std,
        "dynamic_range":  dynamic_range,
        "laplacian_var":  lap_var,
        "dark_ratio":     dark_pixel_ratio,
        "width":          image.width,
        "height":         image.height,
        "aspect_ratio":   image.width / max(image.height, 1),
    }


def classify_page(
    image: Image.Image,
    native_text: Optional[str] = None,
    filename: Optional[str] = None,
) -> PageType:
    """
    Classify a page/image into a PageType using heuristic image analysis.

    Args:
        image:        PIL Image of the rendered page.
        native_text:  Native PDF text (if any was already extracted by PyMuPDF).
                      If provided and substantial, page is classified as DIGITAL_CLEAN.
        filename:     Original document filename for extension/name hints.

    Returns:
        PageType enum value.
    """
    # ── Hint 1: Native text available → definitely digital ─────────────────
    if native_text and len(native_text.strip()) > 50:
        logger.debug("classify_page → DIGITAL_CLEAN (native text present)")
        return PageType.DIGITAL_CLEAN

    # ── Hint 2: Filename/extension hints (CAD drawings, P&IDs) ─────────────
    if filename:
        fn_lower = filename.lower()
        if any(kw in fn_lower for kw in ("pid", "p&id", "schematic", "drawing", "dwg", "dxf", "a3", "a1")):
            logger.debug("classify_page → ENGINEERING_DRAWING (filename hint)")
            return PageType.ENGINEERING_DRAWING

    try:
        stats = _image_stats(image)
    except Exception as exc:
        logger.warning(f"classify_page: image stats failed ({exc}), defaulting to SCANNED_LIGHT")
        return PageType.SCANNED_LIGHT

    w, h = stats["width"], stats["height"]
    aspect = stats["aspect_ratio"]
    dyn    = stats["dynamic_range"]
    std    = stats["std"]
    mean   = stats["mean"]
    dark_r = stats["dark_ratio"]
    lap    = stats["laplacian_var"]

    # ── Hint 3: Large-format landscape → engineering drawing ───────────────
    if aspect > 1.35 and (w > 1800 or h > 1800):
        logger.debug("classify_page → ENGINEERING_DRAWING (large landscape)")
        return PageType.ENGINEERING_DRAWING

    # ── Hint 4: Digital clean — very high dynamic range, sharp, low noise ──
    # A cleanly rendered PDF page typically has very white background (mean > 200)
    # and very crisp dark text (high laplacian sharpness).
    if mean > 210 and dyn > 200 and lap > 300:
        logger.debug("classify_page → DIGITAL_CLEAN (sharp, high-contrast, white background)")
        return PageType.DIGITAL_CLEAN

    # ── Hint 5: Handwritten — medium std, scattered strokes, low dark ratio ─
    # Handwritten pages have irregular stroke widths → lower Laplacian than typed,
    # but not as uniform/flat as a blank page.
    if dark_r < 0.06 and std > 20 and lap < 150:
        logger.debug("classify_page → HANDWRITTEN (sparse irregular strokes)")
        return PageType.HANDWRITTEN

    # ── Hint 6: Heavily degraded — low dynamic range or very low mean ───────
    if dyn < 120 or mean < 80:
        logger.debug("classify_page → SCANNED_HEAVY (low dynamic range or very dark)")
        return PageType.SCANNED_HEAVY

    # ── Hint 7: Noisy faded scan — moderate dynamic range, high noise ───────
    if dyn < 180 or std > 55:
        logger.debug("classify_page → SCANNED_HEAVY (faded/noisy scan)")
        return PageType.SCANNED_HEAVY

    # Default: ordinary office scan, good enough quality
    logger.debug("classify_page → SCANNED_LIGHT (default)")
    return PageType.SCANNED_LIGHT


def get_adaptive_dpi(page_type: PageType) -> int:
    """Return the appropriate render DPI for a given page type."""
    return ADAPTIVE_DPI.get(page_type, 200)


def get_preprocessing_tier(page_type: PageType) -> PreprocessTier:
    """Return the preprocessing tier for a given page type."""
    return ADAPTIVE_PREPROCESS_TIER.get(page_type, PreprocessTier.LIGHT)


def route_document_page(
    image: Image.Image,
    native_text: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full routing decision for a single page/image.

    Returns a routing dict consumed by the OCR pipeline:
    {
        "page_type":    PageType,
        "dpi":          int,
        "tier":         PreprocessTier,
        "skip_ocr":     bool,   # True if native text is sufficient
        "has_handwriting": bool,
    }
    """
    # If native text is sufficient, skip OCR entirely for this page
    from .ocr import is_text_insufficient
    if native_text and not is_text_insufficient(native_text):
        return {
            "page_type":       PageType.DIGITAL_CLEAN,
            "dpi":             150,
            "tier":            PreprocessTier.SKIP,
            "skip_ocr":        True,
            "has_handwriting": False,
        }

    page_type = classify_page(image, native_text=native_text, filename=filename)
    dpi       = get_adaptive_dpi(page_type)
    tier      = get_preprocessing_tier(page_type)

    return {
        "page_type":       page_type,
        "dpi":             dpi,
        "tier":            tier,
        "skip_ocr":        False,
        "has_handwriting": (page_type == PageType.HANDWRITTEN),
    }
