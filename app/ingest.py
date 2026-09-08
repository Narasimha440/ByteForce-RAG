"""
CLI Entry Point for Document Ingestion in SIH 26117.

Usage:
  python -m app.ingest               # Sync all documents in data/
  python -m app.ingest <file_path>   # Ingest a single file
"""

from pathlib import Path
import sys

from .ingestion import IngestionEngine


def main():
    engine = IngestionEngine()

    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg.lower() in ("ocr", "ocr_data"):
            from .config import OCR_DATA_DIR
            print(f"Syncing OCR documents folder: {OCR_DATA_DIR}")
            summary = engine.sync(data_dir=OCR_DATA_DIR)
            print(f"OCR Sync Summary: {summary}")
        elif arg.lower() in ("rag", "rag_data"):
            from .config import RAG_DATA_DIR
            print(f"Syncing RAG documents folder: {RAG_DATA_DIR}")
            summary = engine.sync(data_dir=RAG_DATA_DIR)
            print(f"RAG Sync Summary: {summary}")
        else:
            target_path = Path(arg)
            if target_path.is_dir():
                print(f"Syncing folder: {target_path}")
                summary = engine.sync(data_dir=target_path)
                print(f"Folder Sync Summary: {summary}")
            else:
                print(f"Ingesting single document: {target_path}")
                result = engine.ingest_document(target_path)
                print(f"Result: {result}")
    else:
        engine.sync()


if __name__ == "__main__":
    main()