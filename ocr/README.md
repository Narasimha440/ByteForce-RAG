# Sovereign Local OCR Subsystem (SIH 26117)

**Subsystem Owner**: Harsha Bacham (OCR & Document Understanding)  
**Problem Statement**: Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for Confidential Industrial Work (MRPL)

---

## Architecture Overview

The OCR Subsystem delivers **100% air-gapped, zero-cloud optical character recognition** for scanned refinery inspection reports, maintenance checklists, P&ID drawing legends, and equipment detail sheets.

```
Scanned PDF Page / Image
          │
          ▼
Text Quality Check (is_text_insufficient)
          │
          ▼
In-Memory PDF Page Rendering (PyMuPDF at 200 DPI, zero disk clutter)
          │
          ▼
Local RapidOCR Inference (ONNX Runtime, INT8 CPU/GPU)
          │
          ▼
ISA-5.1 Tag Sanitizer (Corrects PT-l0l -> PT-101, 24 VOC -> 24 VDC)
          │
          ▼
Structured Document Record with Extraction Metadata
(text, confidence, page_number, extraction_method='ocr', ocr_used=True)
```

---

## Key Features

1. **Zero External Binaries**:
   Runs on `rapidocr-onnxruntime`. Does not require system-level Poppler, Tesseract C++ binaries, or administrative installation rights.
2. **In-Memory Rendering**:
   PDF pages are converted directly in RAM via `pymupdf` (`fitz`) at 200 DPI RGB.
3. **ISA-5.1 Instrumentation Tag Sanitizer**:
   Automatically corrects common optical character recognition errors:
   - Replaces optical `O`/`o` with `0` inside equipment tag digits.
   - Replaces optical `l`/`I`/`|` with `1` inside equipment tag digits.
   - Standardizes units: `24 VDC`, `120 VAC`, `230 VAC`, `4-20 mA`, `bar(g)`.
4. **Extraction Confidence**:
   Tracks average line confidence per page, filtering low-confidence noise.

---

## Python API Usage

```python
from ocr import extract_text_with_ocr, get_ocr_engine, sanitize_industrial_tags

# 1. Extract text and confidence from an image or scanned page
text, confidence = extract_text_with_ocr("path/to/scanned_report.png")
print(f"Confidence: {confidence:.2f}")
print(f"Text: {text}")

# 2. Sanitize raw text for industrial tags
raw = "TRANSMITTER PT-l0l POWER 24 VOC SHUTDOWN XV-30l"
clean = sanitize_industrial_tags(raw)
print(clean)  # "TRANSMITTER PT-101 POWER 24 VDC SHUTDOWN XV-301"
```

---

## CLI Usage

```powershell
# Run OCR engine self-check
python -m ocr

# Run OCR on any document or image
python -m ocr path/to/image.png
```
