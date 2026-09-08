"""
Sovereign OCR Subsystem for SIH 26117 (Harsha).

100% local, air-gapped Optical Character Recognition and Technical Document Understanding.
Engine: RapidOCR on ONNX Runtime (zero external C++ binary/Poppler requirements).
Includes ISA-5.1 industrial instrumentation tag error correction.
"""

from app.ocr import (
    BaseOCREngine,
    RapidOCREngine,
    MockOCREngine,
    TesseractOCREngine,
    extract_text_with_ocr,
    get_ocr_engine,
    is_text_insufficient,
    render_pdf_page_to_image,
    sanitize_industrial_tags,
)

__all__ = [
    "BaseOCREngine",
    "RapidOCREngine",
    "MockOCREngine",
    "TesseractOCREngine",
    "extract_text_with_ocr",
    "get_ocr_engine",
    "is_text_insufficient",
    "render_pdf_page_to_image",
    "sanitize_industrial_tags",
]
