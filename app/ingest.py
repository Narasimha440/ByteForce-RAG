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
        target_path = Path(sys.argv[1])
        print(f"Ingesting single document: {target_path}")
        result = engine.ingest_document(target_path)
        print(f"Result: {result}")
    else:
        engine.sync()


if __name__ == "__main__":
    main()