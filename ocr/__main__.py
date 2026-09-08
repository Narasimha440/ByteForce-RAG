"""
CLI Entry Point for the Sovereign OCR Subsystem (Harsha).

Usage:
  python -m ocr <file_or_image_path>
  python -m ocr                         # Run OCR engine self-check
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ocr import extract_text_with_ocr, get_ocr_engine, sanitize_industrial_tags


def main():
    print("==================================================")
    print("SOVEREIGN LOCAL OCR ENGINE (SIH 26117)")
    print("==================================================")

    engine = get_ocr_engine()
    print(f"Active Engine : {type(engine).__name__}")
    print(f"Available     : {engine.is_available()}\n")

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.exists():
            print(f"Error: File not found at {target}")
            sys.exit(1)

        print(f"Processing: {target.name}...")
        try:
            text, conf = extract_text_with_ocr(target)
            print(f"Confidence: {conf:.4f}")
            print("--------------------------------------------------")
            print("Extracted & Sanitized Text:")
            print("--------------------------------------------------")
            print(text[:1000] if text else "(No text detected)")
            if len(text) > 1000:
                print(f"... [truncated, total characters: {len(text)}]")
        except Exception as e:
            print(f"OCR Error: {e}")
            sys.exit(1)
    else:
        # Self-test with sample instrumentation text string
        sample_tag_text = "VALVE PT-l0l PRESSURE 24 VOC SHUTDOWN XV-30l"
        cleaned = sanitize_industrial_tags(sample_tag_text)
        print("Self-Test Tag Sanitizer:")
        print(f"  Raw Input : {sample_tag_text}")
        print(f"  Sanitized : {cleaned}")
        print("\nTip: Run 'python -m ocr <image_path>' to run OCR on a document or image.")

    print("==================================================")


if __name__ == "__main__":
    main()
