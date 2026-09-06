import json
import logging
import random
from pathlib import Path

import requests

from .bm25 import BM25Index
from .config import BM25_INDEX_PATH, DATA_DIR, OLLAMA_GENERATE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_qa.jsonl"


def generate_qa_pairs(num_pairs: int = 10):
    """
    Generate synthetic Q&A pairs from ingested document chunks for domain adaptation
    and fine-tuning of the embedding model/reranker.
    """
    logger.info("Initializing Synthetic Data Generator...")
    
    if not BM25_INDEX_PATH.exists():
        logger.error("No BM25 index found. Please run the ingestion pipeline first.")
        return

    bm25 = BM25Index()
    bm25.load(BM25_INDEX_PATH)
    
    chunks = bm25.doc_payloads
    if not chunks:
        logger.error("No chunks available for synthetic generation.")
        return

    # Filter out very short or empty chunks
    valid_chunks = [c for c in chunks if len(c.get("text", "")) > 200]
    
    if len(valid_chunks) < num_pairs:
        logger.warning(f"Only {len(valid_chunks)} valid chunks available. Adjusting target to {len(valid_chunks)}.")
        num_pairs = len(valid_chunks)

    selected_chunks = random.sample(valid_chunks, num_pairs)
    generated = 0
    
    with open(SYNTHETIC_DATA_PATH, "a", encoding="utf-8") as f:
        for idx, chunk in enumerate(selected_chunks):
            text = chunk.get("text", "")
            
            prompt = (
                "You are an expert industrial engineer creating a training dataset for an AI assistant. "
                "Read the following context and generate a realistic, complex question that an engineer "
                "might ask, and the exact answer based ONLY on the context. "
                "Output strictly in JSON format with 'query' and 'answer' keys.\n\n"
                f"CONTEXT:\n{text}\n\n"
                "JSON OUTPUT:"
            )
            
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.3}
            }
            
            try:
                response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=60)
                response.raise_for_status()
                
                content = response.json().get("response", "").strip()
                if content:
                    qa_data = json.loads(content)
                    
                    if "query" in qa_data and "answer" in qa_data:
                        record = {
                            "query": qa_data["query"],
                            "context": text,
                            "answer": qa_data["answer"],
                            "source_file": chunk.get("filename", "unknown"),
                            "chunk_id": chunk.get("chunk_id", "unknown")
                        }
                        f.write(json.dumps(record) + "\n")
                        generated += 1
                        logger.info(f"Generated pair {generated}/{num_pairs} from {chunk.get('filename')}")
                        
            except Exception as e:
                logger.error(f"Failed to generate pair for chunk {idx}: {e}")
                
    logger.info(f"Synthetic generation complete. Saved {generated} pairs to {SYNTHETIC_DATA_PATH}")

if __name__ == "__main__":
    # Configure logging for standalone script execution
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_qa_pairs(num_pairs=5)
