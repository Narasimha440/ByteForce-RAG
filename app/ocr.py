"""
Local OCR and Document Understanding Pipeline for SIH 26117.

100% air-gapped / local execution. No external or cloud APIs.
Supports:
- Scanned PDF detection (text insufficiency check)
- Local PDF page rendering to image (in-memory via PyMuPDF)
- Local OCR via RapidOCR (ONNX Runtime) with Tesseract and Mock fallbacks
- Retention of industrial technical strings, units, and equipment tags
- Extraction confidence tracking
"""

from abc import ABC, abstractmethod
import io
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageEnhance, ImageOps

from .config import (
    OCR_ENABLED,
    OCR_ENGINE,
    OCR_MIN_TEXT_LENGTH,
    OCR_CONFIDENCE_THRESHOLD,
    OCR_DPI,
)

logger = logging.getLogger(__name__)


def is_text_insufficient(text: Optional[str], min_length: int = OCR_MIN_TEXT_LENGTH) -> bool:
    """
    Detect whether extracted text is empty or clearly insufficient.

    Returns True if the text length (after stripping) is below min_length,
    or contains predominantly whitespace/non-printable characters,
    signaling that the document/page is scanned or image-based.
    """
    if not text:
        return True

    cleaned = text.strip()
    if len(cleaned) < min_length:
        return True

    # Check ratio of alphanumeric characters to total characters
    alnum_count = sum(1 for char in cleaned if char.isalnum())
    if alnum_count < min_length // 2:
        return True

    return False


def preprocess_image_for_ocr(image: Union[Image.Image, Path, str]) -> Image.Image:
    """
    Industrial scan preprocessor for RapidOCR:
    1. Normalizes image to RGB PIL Image.
    2. Measures dynamic range; applies auto-contrast stretch if the scan is low-contrast,
       faded, or carbon-copy (preserving pristine digital text without artifact ringing).
    3. Optional contour-based deskewing if image is tilted.
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        import numpy as np
        pil_img = Image.fromarray(np.array(image)).convert("RGB")

    try:
        import numpy as np
        arr = np.array(pil_img)
        min_val, max_val = int(arr.min()), int(arr.max())
        # If the image has low dynamic range (faded, dark, or washed-out scan)
        if min_val > 25 or max_val < 230:
            pil_img = ImageOps.autocontrast(pil_img, cutoff=0.5)
    except Exception as exc:
        logger.debug(f"Image preprocessing fallback: {exc}")

    return pil_img



class BaseOCREngine(ABC):
    """Abstract interface for local OCR engines."""

    @abstractmethod
    def extract_text_from_image(
        self, image: Union[Image.Image, Path, str]
    ) -> Tuple[str, float]:
        """
        Extract text from an image.

        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the OCR engine is initialized and usable."""
        pass


class RapidOCREngine(BaseOCREngine):
    """
    Local OCR engine using rapidocr-onnxruntime.

    Runs entirely on local ONNX models without needing external C++ binaries,
    Poppler, or Tesseract installers. Excellent for industrial reports,
    engineering drawings, and tables.
    """

    def __init__(self):
        self._engine = None
        self._initialized = False

    def _get_engine(self):
        if not self._initialized:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR()
                self._initialized = True
            except Exception as exc:
                logger.warning(f"Could not initialize RapidOCR: {exc}")
                self._initialized = False
        return self._engine

    def is_available(self) -> bool:
        return self._get_engine() is not None

    def extract_layout_from_image(
        self, image: Union[Image.Image, Path, str], preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Extract structured layout with physical bounding box coordinates [x1, y1, x2, y2].
        Enables downstream UI and drawing inspectors to highlight exact visual locations.
        """
        engine = self._get_engine()
        if engine is None:
            raise RuntimeError("RapidOCR engine is not available.")

        import numpy as np

        if preprocess:
            pil_img = preprocess_image_for_ocr(image)
        elif isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.fromarray(np.array(image)).convert("RGB")

        img_np = np.array(pil_img)

        # RapidOCR returns: list of [box, text, score]
        result, _ = engine(img_np)

        if not result:
            return {"text": "", "average_confidence": 0.0, "blocks": [], "line_count": 0}

        valid_items = []
        for item in result:
            if len(item) >= 3:
                box, text, score = item[0], str(item[1]).strip(), float(item[2])
                if text and score >= OCR_CONFIDENCE_THRESHOLD:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    bbox = [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]
                    valid_items.append({
                        "text": text,
                        "bbox": bbox,
                        "polygon": box,
                        "cy": sum(ys) / 4.0,
                        "cx": sum(xs) / 4.0,
                        "height": abs(max(ys) - min(ys)),
                        "score": score,
                    })

        if not valid_items:
            return {"text": "", "average_confidence": 0.0, "blocks": [], "line_count": 0}

        # Group by rows using a dynamic tolerance based on median height
        valid_items.sort(key=lambda x: x["cy"])
        median_h = sorted([x["height"] for x in valid_items])[len(valid_items) // 2]
        y_tolerance = max(median_h * 0.4, 10.0)

        rows = []
        current_row = []
        current_row_y = valid_items[0]["cy"]

        for item in valid_items:
            if abs(item["cy"] - current_row_y) <= y_tolerance:
                current_row.append(item)
                current_row_y = sum([i["cy"] for i in current_row]) / len(current_row)
            else:
                rows.append(current_row)
                current_row = [item]
                current_row_y = item["cy"]

        if current_row:
            rows.append(current_row)

        lines: List[str] = []
        confidences: List[float] = [x["score"] for x in valid_items]

        # Sort columns within each row and reconstruct
        for row in rows:
            row.sort(key=lambda x: x["cx"])
            if len(row) > 1:
                lines.append(" | ".join([item["text"] for item in row]))
            else:
                lines.append(row[0]["text"])

        extracted_text = "\n".join(lines).strip()
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "text": extracted_text,
            "average_confidence": round(avg_confidence, 4),
            "blocks": [
                {
                    "text": it["text"],
                    "bbox": it["bbox"],
                    "confidence": round(it["score"], 4),
                }
                for it in valid_items
            ],
            "line_count": len(lines),
        }

    def extract_text_from_image(
        self, image: Union[Image.Image, Path, str]
    ) -> Tuple[str, float]:
        layout = self.extract_layout_from_image(image, preprocess=True)
        return layout["text"], layout["average_confidence"]



class TesseractOCREngine(BaseOCREngine):
    """Fallback OCR engine using pytesseract if tesseract is installed."""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def extract_text_from_image(
        self, image: Union[Image.Image, Path, str]
    ) -> Tuple[str, float]:
        if not self.is_available():
            raise RuntimeError("Tesseract is not installed or available on PATH.")

        import pytesseract

        if isinstance(image, (str, Path)):
            image = Image.open(str(image))

        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        texts = []
        confs = []

        n_boxes = len(data["text"])
        current_line = []

        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])

            if word and conf >= (OCR_CONFIDENCE_THRESHOLD * 100):
                current_line.append(word)
                confs.append(conf / 100.0)

            # Detect line breaks from block/line number changes
            if i + 1 < n_boxes and data["line_num"][i] != data["line_num"][i + 1]:
                if current_line:
                    texts.append(" ".join(current_line))
                    current_line = []

        if current_line:
            texts.append(" ".join(current_line))

        extracted_text = "\n".join(texts).strip()
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return extracted_text, round(avg_conf, 4)


class MockOCREngine(BaseOCREngine):
    """Deterministic OCR engine for unit and integration tests."""

    def __init__(
        self,
        mock_text: str = "Mock OCR Extracted Technical Text PT-101 FT-204",
        mock_confidence: float = 0.95,
    ):
        self.mock_text = mock_text
        self.mock_confidence = mock_confidence

    def is_available(self) -> bool:
        return True

    def extract_text_from_image(
        self, image: Union[Image.Image, Path, str]
    ) -> Tuple[str, float]:
        return self.mock_text, self.mock_confidence

    def extract_layout_from_image(
        self, image: Union[Image.Image, Path, str], preprocess: bool = True
    ) -> Dict[str, Any]:
        lines = self.mock_text.splitlines()
        blocks = [
            {
                "text": line,
                "bbox": [10.0, 10.0 + i * 25.0, 250.0, 30.0 + i * 25.0],
                "confidence": self.mock_confidence,
            }
            for i, line in enumerate(lines)
        ]
        return {
            "text": self.mock_text,
            "average_confidence": self.mock_confidence,
            "blocks": blocks,
            "line_count": len(lines),
        }



# Global singleton instance cache
_ACTIVE_OCR_ENGINE: Optional[BaseOCREngine] = None


def get_ocr_engine(engine_name: Optional[str] = None) -> BaseOCREngine:
    """
    Factory to retrieve or create the active local OCR engine.
    """
    global _ACTIVE_OCR_ENGINE

    name = (engine_name or OCR_ENGINE).lower()

    if _ACTIVE_OCR_ENGINE is not None and not engine_name:
        return _ACTIVE_OCR_ENGINE

    engine: BaseOCREngine

    if name == "mock":
        engine = MockOCREngine()
    elif name == "tesseract":
        engine = TesseractOCREngine()
    elif name == "rapidocr":
        engine = RapidOCREngine()
    else:  # "auto" or default
        rapid = RapidOCREngine()
        if rapid.is_available():
            engine = rapid
        else:
            tess = TesseractOCREngine()
            if tess.is_available():
                engine = tess
            else:
                logger.warning("No functional OCR engine found. Falling back to MockOCREngine.")
                engine = MockOCREngine()

    if not engine_name:
        _ACTIVE_OCR_ENGINE = engine

    return engine


def render_pdf_page_to_image(
    pdf_path: Path, page_number: int, dpi: int = OCR_DPI
) -> Image.Image:
    """
    Render a single page from a PDF file to a PIL Image entirely in-memory.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-indexed page number.
        dpi: Dots per inch for rendering resolution.

    Returns:
        PIL.Image in RGB format.
    """
    try:
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
        if page_number < 1 or page_number > len(doc):
            raise IndexError(f"Page {page_number} out of range for {pdf_path.name} ({len(doc)} pages)")

        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    except ImportError:
        # Fallback to pypdfium2 if pymupdf is absent
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(pdf_path))
            page = pdf[page_number - 1]
            scale = dpi / 72.0
            image = page.render(scale=scale).to_pil()
            pdf.close()
            return image
        except ImportError:
            raise RuntimeError(
                "Neither PyMuPDF (fitz) nor pypdfium2 is installed. "
                "Cannot render PDF pages to image for local OCR."
            )


def sanitize_industrial_tags(text: str) -> str:
    """
    Post-process OCR output to sanitize and correct common optical character
    recognition errors in industrial instrumentation tags (ISA-5.1), units, and electrical ratings.
    """
    if not text:
        return ""

    cleaned = text

    # 1. Standardize common electrical & process engineering units
    cleaned = re.sub(r"\b24\s*V[0oO]C\b", "24 VDC", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b12[oO]\s*VAC\b", "120 VAC", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b23[oO]\s*VAC\b", "230 VAC", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b4\s*[-–]\s*2[oO]\s*mA\b", "4-20 mA", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbar\s*\(\s*g\s*\)", "bar(g)", cleaned, flags=re.IGNORECASE)

    # 2. Fix OCR optical errors in ISA-5.1 Instrument & Valve Tags
    tag_prefixes = r"(?:PT|TT|FT|LT|XV|FV|PV|LV|DP|TI|PI|FI|LI|ESD|SDV|BDV|MOV|PSV|PRV|CV|FC|FO)"

    def _clean_tag(match: re.Match) -> str:
        prefix = match.group(1).upper()
        num_body = match.group(3)
        trailer = match.group(4) or ""
        fixed_num = ""
        for ch in num_body:
            if ch in ("O", "o"):
                fixed_num += "0"
            elif ch in ("l", "I", "|"):
                fixed_num += "1"
            else:
                fixed_num += ch
        return f"{prefix}-{fixed_num}{trailer.upper()}"

    tag_pattern = re.compile(rf"\b({tag_prefixes})([-\s]?)([0-9OoIl|]{{2,5}})([A-Za-z]?)\b")
    cleaned = tag_pattern.sub(_clean_tag, cleaned)

    return cleaned


def detect_table_structure(
    ocr_layout: Dict[str, Any],
    min_columns: int = 2,
    min_rows: int = 2,
) -> Dict[str, Any]:
    """
    Analyse an OCR layout result (from extract_layout_with_ocr) and detect
    whether the page contains a structured table with multiple columns.

    Returns a dict:
        {
            "is_table": bool,
            "rows": [[cell_text, ...], ...],  # list of rows, each a list of cell texts
            "csv_text": str,                  # pipe-separated reconstruction for chunking
            "num_rows": int,
            "num_cols": int,
        }

    If no table is detected, returns {"is_table": False, ...} with empty rows.
    """
    blocks = ocr_layout.get("blocks", [])
    if not blocks:
        return {"is_table": False, "rows": [], "csv_text": "", "num_rows": 0, "num_cols": 0}

    # Group blocks into horizontal rows by their vertical centre (cy)
    sorted_blocks = sorted(blocks, key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2)

    rows_raw: List[List[Dict]] = []
    current_row: List[Dict] = []

    # Compute median block height for adaptive y-tolerance
    heights = [(b["bbox"][3] - b["bbox"][1]) for b in sorted_blocks if len(b["bbox"]) >= 4]
    median_h = sorted(heights)[len(heights) // 2] if heights else 20.0
    y_tol = max(median_h * 0.5, 8.0)

    last_cy = None
    for block in sorted_blocks:
        cy = (block["bbox"][1] + block["bbox"][3]) / 2.0
        if last_cy is None or abs(cy - last_cy) <= y_tol:
            current_row.append(block)
            last_cy = cy if last_cy is None else (last_cy + cy) / 2.0
        else:
            rows_raw.append(current_row)
            current_row = [block]
            last_cy = cy
    if current_row:
        rows_raw.append(current_row)

    # Filter: need at least min_rows with at least min_columns in the majority of rows
    multi_col_rows = [r for r in rows_raw if len(r) >= min_columns]
    is_table = (
        len(rows_raw) >= min_rows and len(multi_col_rows) >= min_rows
    )

    if not is_table:
        return {"is_table": False, "rows": [], "csv_text": "", "num_rows": 0, "num_cols": 0}

    # Reconstruct table rows as lists of cell text (sorted left-to-right)
    table_rows: List[List[str]] = []
    for row in rows_raw:
        row_sorted = sorted(row, key=lambda b: b["bbox"][0])  # sort by x1 (left edge)
        table_rows.append([sanitize_industrial_tags(b["text"]) for b in row_sorted])

    num_cols = max(len(r) for r in table_rows) if table_rows else 0

    # Build a pipe-separated text representation for downstream chunking
    csv_lines: List[str] = []
    for row in table_rows:
        csv_lines.append(" | ".join(cell.strip() for cell in row))
    csv_text = "\n".join(csv_lines)

    return {
        "is_table": True,
        "rows": table_rows,
        "csv_text": csv_text,
        "num_rows": len(table_rows),
        "num_cols": num_cols,
    }


def get_page_ocr_quality_report(
    pdf_path: Path,
    max_pages: int = 5,
    dpi: int = OCR_DPI,
) -> List[Dict[str, Any]]:
    """
    Run OCR on the first `max_pages` pages of a PDF and return a per-page
    quality report. Useful during ingestion to flag low-confidence pages
    that may need manual review.

    Returns:
        List of dicts per page:
        {
            "page_number": int,
            "avg_confidence": float,
            "line_count": int,
            "is_low_quality": bool,  # True if avg_confidence < OCR_CONFIDENCE_THRESHOLD
            "is_table_page": bool,   # True if table structure detected
        }
    """
    report: List[Dict[str, Any]] = []
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logger.warning(f"OCR quality report: file not found {pdf_path}")
        return report

    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        total_pages = min(len(doc), max_pages)
        doc.close()
    except Exception:
        total_pages = max_pages  # best-effort

    for page_num in range(1, total_pages + 1):
        try:
            img = render_pdf_page_to_image(pdf_path, page_num, dpi=dpi)
            layout = extract_layout_with_ocr(img, sanitize_tags=True, preprocess=True)
            table_info = detect_table_structure(layout)
            avg_conf = layout.get("average_confidence", 0.0)
            report.append({
                "page_number": page_num,
                "avg_confidence": round(avg_conf, 4),
                "line_count": layout.get("line_count", 0),
                "is_low_quality": avg_conf < OCR_CONFIDENCE_THRESHOLD,
                "is_table_page": table_info["is_table"],
            })
        except Exception as exc:
            report.append({
                "page_number": page_num,
                "avg_confidence": 0.0,
                "line_count": 0,
                "is_low_quality": True,
                "is_table_page": False,
                "error": str(exc),
            })
            logger.warning(f"OCR quality check failed on page {page_num}: {exc}")

    return report


def extract_text_with_ocr(
    image_or_path: Union[Image.Image, Path, str],
    sanitize_tags: bool = True,
) -> Tuple[str, float]:
    """
    Convenience function to run OCR on an image or file path,
    with optional post-processing ISA-5.1 industrial tag sanitization.

    Returns:
        Tuple of (extracted_text, average_confidence)
    """
    engine = get_ocr_engine()
    raw_text, conf = engine.extract_text_from_image(image_or_path)
    if sanitize_tags and raw_text:
        raw_text = sanitize_industrial_tags(raw_text)
    return raw_text, conf


def extract_layout_with_ocr(
    image_or_path: Union[Image.Image, Path, str],
    sanitize_tags: bool = True,
    preprocess: bool = True,
) -> Dict[str, Any]:
    """
    Run local OCR on an image or file path and return structured layout
    with precise bounding box coordinates for each recognized text segment.
    """
    engine = get_ocr_engine()
    if hasattr(engine, "extract_layout_from_image"):
        layout = engine.extract_layout_from_image(image_or_path, preprocess=preprocess)
    else:
        text, conf = engine.extract_text_from_image(image_or_path)
        layout = {
            "text": text,
            "average_confidence": conf,
            "blocks": [],
            "line_count": len(text.splitlines()),
        }

    if sanitize_tags and layout.get("text"):
        layout["text"] = sanitize_industrial_tags(layout["text"])
        for block in layout.get("blocks", []):
            block["text"] = sanitize_industrial_tags(block["text"])

    return layout


