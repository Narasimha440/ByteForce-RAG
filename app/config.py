from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OCR_DATA_DIR = DATA_DIR / "ocr"
RAG_DATA_DIR = DATA_DIR / "rag"

# Vector Database (Qdrant)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sih26117_knowledge"
QDRANT_STORAGE_PATH = DATA_DIR / "qdrant_storage"
QDRANT_PREFER_LOCAL = False  # If True or if QDRANT_URL is unreachable, use embedded on-disk storage

# Local LLM (Ollama)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# Preferred Lightweight Open-Source Local Models (MRPL Enterprise Grade)
PREFERRED_OLLAMA_MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "qwen2.5:7b-instruct",
    "qwen2.5:7b",
    "deepseek-r1:8b",
    "llama3.2:3b",
    "llama3.2:1b",
    "mistral:7b",
    "qwen2.5-coder:7b",
]

def get_active_ollama_model(default_model: str = "qwen2.5:1.5b") -> str:
    """
    Dynamically discover installed models in local Ollama service.
    Returns the highest-priority installed model matching PREFERRED_OLLAMA_MODELS,
    or the first available installed model, or default_model.
    """
    import requests
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=1.5)
        if resp.ok:
            models_data = resp.json().get("models", [])
            installed = [m.get("name", "") for m in models_data if m.get("name")]
            if installed:
                for pref in PREFERRED_OLLAMA_MODELS:
                    for inst in installed:
                        if pref == inst or pref in inst or inst.startswith(pref.split(":")[0]):
                            return inst
                return installed[0]
    except Exception:
        pass
    return default_model

OLLAMA_MODEL = get_active_ollama_model()
OLLAMA_VISION_MODEL = "llava"  # or minicpm-v

# Embedding Provider
EMBEDDING_MODEL = "BAAI/bge-m3" 
EMBEDDING_PROVIDER = "local"  # Using local sentence_transformers as before
EMBEDDING_API_URL = "http://localhost:8000/embed"

# Chunking & Splitting
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 80

# Hybrid Retrieval & Search Quality
TOP_K = 4                    # Reduced from 5 — fewer candidates = faster CrossEncoder
SIMILARITY_THRESHOLD = 0.38  # Raised from 0.35 — skips weak candidates earlier
HYBRID_RETRIEVAL_ENABLED = True
BM25_INDEX_PATH = DATA_DIR / "bm25_index.json"
RRF_K = 40                   # Reduced from 60 — slightly faster RRF computation

# LLM Context Window Limit
# Prevents sending huge context strings that slow Ollama token generation
LLM_MAX_CONTEXT_CHARS = 6000  # Truncate combined evidence context to this length

# Reranker Settings
RERANKING_ENABLED = False  # Disabled to massively speed up search latency on CPU
RERANKER_ENGINE = "cross_encoder"  # P1 FIX: Mandated Cross-Encoder for deep attention
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Local OCR Settings
OCR_ENABLED = True
OCR_ENGINE = "rapidocr"  # "rapidocr", "tesseract", "auto", "mock"
OCR_LANGUAGE = "en"
OCR_MIN_TEXT_LENGTH = 50  # Characters below which a page is treated as scanned/insufficient
OCR_CONFIDENCE_THRESHOLD = 0.65  # Raised from 0.50 — filters low-confidence OCR noise
OCR_DPI = 250  # Fallback DPI for the quality reporting tool; adaptive DPI is now used during ingestion
OCR_TABLE_DETECTION_ENABLED = True  # Detect and preserve table structure during OCR