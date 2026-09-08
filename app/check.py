"""
System Diagnostics and Health Checks for SIH 26117.

Checks:
1. Qdrant (Server and embedded on-disk mode)
2. Ollama (Model readiness and Qwen3 8B availability)
3. Local OCR Engine (RapidOCR / ONNX Runtime / PyMuPDF)
4. Embedding Provider (BGE-M3 / API Provider)
"""

import requests

from .config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OCR_ENABLED,
    OCR_ENGINE,
    OLLAMA_MODEL,
    OLLAMA_URL,
    QDRANT_STORAGE_PATH,
    QDRANT_URL,
)


def check_qdrant():
    print("=" * 60)
    print("Checking Qdrant...")
    print("=" * 60)

    server_ok = False
    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=3)
        if response.ok:
            print(f"Qdrant Server: OK ({QDRANT_URL})")
            server_ok = True
        else:
            print(f"Qdrant Server: HTTP {response.status_code}")
    except Exception:
        print(f"Qdrant Server ({QDRANT_URL}): OFFLINE / NOT RUNNING")

    # Check local embedded storage fallback
    print(f"Embedded Storage Fallback Path: {QDRANT_STORAGE_PATH}")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
        cols = client.get_collections()
        print("Embedded Local Qdrant: READY (zero-Docker standalone operational)")
    except Exception as err:
        print(f"Embedded Local Qdrant check: {err}")


def check_ollama():
    print()
    print("=" * 60)
    print("Checking Ollama...")
    print("=" * 60)

    try:
        base_url = OLLAMA_URL.rsplit("/api", 1)[0]
        response = requests.get(f"{base_url}/api/tags", timeout=5)

        if not response.ok:
            print(f"Ollama: ERROR {response.status_code}")
            return

        data = response.json()
        models = [model.get("name") for model in data.get("models", [])]

        print("Ollama Service: OK")
        print(f"Installed models ({len(models)}):")
        for m in models:
            print(f"  - {m}")

        if any(OLLAMA_MODEL in m for m in models):
            print(f"Target LLM: FOUND ({OLLAMA_MODEL})")
        else:
            print(f"Target LLM: NOT FOUND ({OLLAMA_MODEL})")
            print(f"  Run 'ollama pull {OLLAMA_MODEL}' to install.")

    except Exception as error:
        print("Ollama Service: OFFLINE / UNREACHABLE")
        print(f"  Details: {error}")


def check_ocr():
    print()
    print("=" * 60)
    print("Checking Local OCR Pipeline...")
    print("=" * 60)

    print(f"OCR Enabled: {OCR_ENABLED}")
    print(f"Configured Engine: {OCR_ENGINE}")

    try:
        from .ocr import get_ocr_engine
        engine = get_ocr_engine()
        available = engine.is_available()
        engine_type = type(engine).__name__
        print(f"Active OCR Engine: {engine_type}")
        print(f"Engine Status: {'READY' if available else 'NOT AVAILABLE'}")
    except Exception as exc:
        print(f"OCR Engine Check: FAILED ({exc})")

    # Check PDF rendering library
    try:
        import pymupdf
        print("PDF Renderer (PyMuPDF): READY")
    except ImportError:
        try:
            import pypdfium2
            print("PDF Renderer (pypdfium2): READY")
        except ImportError:
            print("PDF Renderer: NEITHER PyMuPDF NOR pypdfium2 FOUND")


def check_embedding_provider():
    print()
    print("=" * 60)
    print("Checking Embedding Provider...")
    print("=" * 60)

    print(f"Provider: {EMBEDDING_PROVIDER}")
    print(f"Model: {EMBEDDING_MODEL}")

    try:
        from .embeddings import get_embedding_provider
        provider = get_embedding_provider()
        dim = provider.get_dimension()
        print(f"Embedding Dimension: {dim}")
        print("Embedding Provider: READY")
    except Exception as exc:
        print(f"Embedding Provider Check: FAILED ({exc})")


def run_all_checks():
    check_qdrant()
    check_ollama()
    check_ocr()
    check_embedding_provider()
    print("=" * 60)


if __name__ == "__main__":
    run_all_checks()