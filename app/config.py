from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

# Vector Database (Qdrant)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sih26117_knowledge"
QDRANT_STORAGE_PATH = DATA_DIR / "qdrant_storage"
QDRANT_PREFER_LOCAL = False  # If True or if QDRANT_URL is unreachable, use embedded on-disk storage

# Local LLM (Ollama)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_VISION_MODEL = "llava"  # or minicpm-v

# Embedding Provider
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_PROVIDER = "local"  # "local" (in-process BGE-M3) or "api" (future internal HTTP API)
EMBEDDING_API_URL = "http://localhost:8000/embed"

# Chunking & Splitting
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 80

# Hybrid Retrieval & Search Quality
TOP_K = 5
SIMILARITY_THRESHOLD = 0.35
HYBRID_RETRIEVAL_ENABLED = True
BM25_INDEX_PATH = DATA_DIR / "bm25_index.json"
RRF_K = 60  # Reciprocal Rank Fusion constant

# Reranker Settings
RERANKING_ENABLED = True
RERANKER_ENGINE = "lexical"  # "lexical" (fast, zero extra VRAM) or "cross_encoder" (deep attention)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Local OCR Settings
OCR_ENABLED = True
OCR_ENGINE = "rapidocr"  # "rapidocr", "tesseract", "auto", "mock"
OCR_LANGUAGE = "en"
OCR_MIN_TEXT_LENGTH = 50  # Characters below which a page is treated as scanned/insufficient
OCR_CONFIDENCE_THRESHOLD = 0.5  # Filter out low-confidence OCR lines
OCR_DPI = 200  # Resolution for rendering PDF pages to image