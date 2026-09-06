"""
Local BM25 Lexical Search Engine for SIH 26117.

Provides fast, exact-match keyword and technical tag retrieval
(e.g., PT-101, FT-204, XV-301, OISD-156, API-570, MAH, ESD) to complement
dense semantic embeddings in a Hybrid Search architecture.
"""

import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def tokenize_industrial_text(text: str) -> List[str]:
    """
    Tokenize technical text while strictly preserving industrial tags,
    equipment codes (e.g. PT-101, XV-301), units, and numbers.
    """
    if not text:
        return []

    # Preserve hyphenated tags, alphanumeric tags, and words
    # Matches words, tags like PT-101, numbers, and units
    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text.lower())
    return tokens


class BM25Index:
    """
    In-memory Okapi BM25 index with persistence.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: List[int] = []
        self.doc_ids: List[str] = []
        self.doc_payloads: List[Dict[str, Any]] = []
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> list of (doc_idx, tf)
        self.idf: Dict[str, float] = {}

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 index from document chunks.
        """
        self.doc_ids = []
        self.doc_payloads = []
        self.doc_lengths = []
        self.inverted_index = {}
        self.idf = {}

        self.corpus_size = len(chunks)
        if self.corpus_size == 0:
            self.avgdl = 0.0
            return

        total_length = 0

        for doc_idx, chunk in enumerate(chunks):
            # Include chunk_id, text, and extracted tags
            tags_text = " ".join(chunk.get("tags", []))
            combined_text = f"{chunk.get('text', '')} {tags_text} {chunk.get('filename', '')} {chunk.get('section', '')}"
            tokens = tokenize_industrial_text(combined_text)

            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # Stable identifier
            cid = chunk.get("chunk_id", str(doc_idx))
            self.doc_ids.append(cid)
            self.doc_payloads.append(chunk)

            # Count term frequencies
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, tf in tf_map.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, tf))

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate IDF for all indexed terms
        for term, postings in self.inverted_index.items():
            df = len(postings)
            # Standard Lucene/Okapi smoothed IDF
            self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """
        Score documents against query terms using Okapi BM25.
        """
        query_tokens = tokenize_industrial_text(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            term_idf = self.idf.get(term, 0.0)
            for doc_idx, tf in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += term_idf * (numerator / denominator)

        # Collect results with positive scores
        scored_docs = [
            (self.doc_payloads[i], scores[i])
            for i in range(self.corpus_size)
            if scores[i] > 0.0
        ]

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def save(self, file_path: Path):
        """Persist the BM25 index to a JSON file."""
        data = {
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,
            "doc_ids": self.doc_ids,
            "doc_payloads": self.doc_payloads,
            "inverted_index": self.inverted_index,
            "idf": self.idf,
        }
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, file_path: Path):
        """Load the BM25 index from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.corpus_size = data["corpus_size"]
        self.avgdl = data["avgdl"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_ids = data["doc_ids"]
        self.doc_payloads = data["doc_payloads"]
        self.inverted_index = {k: [tuple(p) for p in v] for k, v in data["inverted_index"].items()}
        self.idf = data["idf"]
