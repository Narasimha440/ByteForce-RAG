"""
Unified CLI for ByteForce-RAG (SIH 26117).

Quick Commands:
  python cli.py rag "<question>"     # Search & retrieve from local RAG engine
  python cli.py ocr [file_path]      # Run OCR on an image or scanned document
  python cli.py ingest               # Ingest & sync knowledge base in data/
  python cli.py check                # Run system & model health checks
  python cli.py test                 # Run automated unit test suite
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def print_banner():
    print("""
======================================================================
 BYTEFORCE-RAG: SOVEREIGN INDUSTRIAL WORKBENCH (SIH 26117)
 MRPL On-Premise Air-Gapped RAG & Local OCR Engine
======================================================================
""")


def print_help():
    print_banner()
    print("""Available CLI Commands:

1. RAG Query (Retrieval & Grounded Context):
   python cli.py rag "What is the trip limit for PT-101?"
   python cli.py rag "What were the safety audit findings for XV-301?"

2. OCR Engine (Local Extraction & Tag Sanitizer):
   python cli.py ocr
   python cli.py ocr path/to/drawing_or_scanned_page.png

3. Ingestion & Document Synchronization:
   python cli.py ingest              # Sync all files in data/
   python cli.py ingest <file_path>  # Ingest a single document

4. Health Check & Diagnostics:
   python cli.py check               # Check Qdrant, Ollama, RapidOCR, BGE-M3

5. Test Suite:
   python cli.py test                # Run unit and integration tests

Alternative Direct Commands:
   python ocr                        # Run OCR self-test
   python ocr <image_path>           # Run OCR on file
   python rag "<question>"           # Run RAG query directly
   python -m app.ingest              # Ingest knowledge base
======================================================================
""")


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    subcommand = sys.argv[1].lower().strip()
    args = sys.argv[2:]

    if subcommand == "rag":
        from rag.__main__ import main as rag_main
        sys.argv = ["rag"] + args
        rag_main()

    elif subcommand == "ocr":
        from ocr.__main__ import main as ocr_main
        sys.argv = ["ocr"] + args
        ocr_main()

    elif subcommand == "ingest":
        from app.ingest import main as ingest_main
        sys.argv = ["ingest"] + args
        ingest_main()

    elif subcommand == "check":
        from app.check import run_all_checks
        run_all_checks()

    elif subcommand == "test":
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

    else:
        print(f"Unknown command: '{subcommand}'")
        print_help()


if __name__ == "__main__":
    main()
