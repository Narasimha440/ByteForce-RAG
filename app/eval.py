"""
RAG Accuracy Evaluation Benchmark for SIH 26117.

Zero external dependencies. Implements lightweight RAGAS-inspired metrics:
    - Hit Rate @ K  : Was the answer found in the top-K retrieved chunks?
    - MRR           : Mean Reciprocal Rank (position of first correct hit)
    - Precision @ K : Fraction of top-K chunks that are relevant

Usage:
    python cli.py rag --eval
    python -m app.eval

Requires:
    data/eval_questions.json  — list of {question, expected_keywords} dicts
    A populated Qdrant + BM25 index (run `python cli.py ingest` first)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_EVAL_FILE = DATA_DIR / "eval_questions.json"
DEFAULT_RESULTS_FILE = DATA_DIR / "eval_results.json"


# ──────────────────────────────────────────────
# Default MRPL evaluation questions
# (generated from the indexed knowledge base)
# ──────────────────────────────────────────────
DEFAULT_EVAL_QUESTIONS: List[Dict] = [
    {
        "question": "What is the installed refining capacity of MRPL Phase III?",
        "expected_keywords": ["MMTPA", "capacity", "Phase III", "refinery"],
        "category": "facility_overview",
    },
    {
        "question": "What is the trip limit or high-high pressure set point for PT-101?",
        "expected_keywords": ["PT-101", "pressure", "trip", "bar", "set point"],
        "category": "instrument_setpoints",
    },
    {
        "question": "What were the key safety audit findings for MRPL in 2026?",
        "expected_keywords": ["audit", "finding", "safety", "2026", "MRPL"],
        "category": "safety_audit",
    },
    {
        "question": "What is the emergency shutdown procedure for the CDU unit?",
        "expected_keywords": ["ESD", "emergency", "shutdown", "CDU", "procedure"],
        "category": "emergency_procedures",
    },
    {
        "question": "What is the stroke time or closing time specification for XV-301?",
        "expected_keywords": ["XV-301", "stroke", "closing time", "seconds", "valve"],
        "category": "instrument_setpoints",
    },
    {
        "question": "What are the OISD standards applicable to MRPL operations?",
        "expected_keywords": ["OISD", "standard", "petroleum", "safety"],
        "category": "standards",
    },
    {
        "question": "What are the major hazards identified in the MAH factory report?",
        "expected_keywords": ["MAH", "hazard", "major", "factory", "accident"],
        "category": "safety_audit",
    },
    {
        "question": "What is the design pressure or maximum allowable working pressure for C-101?",
        "expected_keywords": ["C-101", "pressure", "design", "MAWP", "bar"],
        "category": "equipment_specs",
    },
    {
        "question": "What is the sulfur content specification for diesel product at MRPL?",
        "expected_keywords": ["sulfur", "diesel", "ppm", "specification", "product"],
        "category": "product_specs",
    },
    {
        "question": "What are the fire safety and emergency response provisions at MRPL?",
        "expected_keywords": ["fire", "emergency", "response", "safety", "provisions"],
        "category": "safety_procedures",
    },
]


def _load_eval_questions(eval_file: Path) -> List[Dict]:
    """Load eval questions from JSON file, or return defaults if file missing."""
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
            logger.info(f"Loaded {len(questions)} eval questions from {eval_file.name}")
            return questions
        except Exception as exc:
            logger.warning(f"Could not load eval file {eval_file}: {exc}. Using defaults.")

    # Write defaults to disk for future customization
    try:
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_EVAL_QUESTIONS, f, indent=2)
        logger.info(f"Created default eval_questions.json at {eval_file}")
    except Exception:
        pass

    return DEFAULT_EVAL_QUESTIONS


def _hit_in_chunk(chunk: Dict, keywords: List[str]) -> bool:
    """
    Return True if any expected keyword appears in the chunk text or tags.
    Case-insensitive, space-insensitive partial match.
    """
    text = (chunk.get("text", "") + " " + " ".join(chunk.get("tags", []))).lower()
    for kw in keywords:
        if kw.lower() in text:
            return True
    return False


class RAGEvaluator:
    """
    Lightweight RAGAS-inspired evaluator for the LocalRAG pipeline.

    Metrics computed:
        hit_rate_at_k  : fraction of questions where a relevant chunk appears in top-K
        mrr            : mean reciprocal rank of first relevant hit
        precision_at_k : average fraction of top-K chunks that are relevant
    """

    def __init__(self, rag=None, top_k: int = 5):
        self.top_k = top_k
        self._rag = rag

    def _get_rag(self):
        if self._rag is None:
            from app.rag import LocalRAG
            self._rag = LocalRAG()
        return self._rag

    def evaluate_single(self, question: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Run retrieval for one question and compute per-question metrics."""
        rag = self._get_rag()
        t0 = time.time()

        try:
            hits = rag.retrieve(question, top_k=self.top_k)
        except Exception as exc:
            logger.error(f"Retrieval failed for '{question}': {exc}")
            return {
                "question": question,
                "error": str(exc),
                "hit": False,
                "reciprocal_rank": 0.0,
                "precision": 0.0,
                "latency_ms": round((time.time() - t0) * 1000),
                "chunks_retrieved": 0,
            }

        latency_ms = round((time.time() - t0) * 1000)

        # Build flat chunk dicts for scoring
        chunks = []
        for h in hits:
            payload = h.payload if hasattr(h, "payload") and h.payload else (h if isinstance(h, dict) else {})
            chunks.append(payload)

        # Reciprocal Rank
        reciprocal_rank = 0.0
        hit = False
        for rank, chunk in enumerate(chunks, start=1):
            if _hit_in_chunk(chunk, expected_keywords):
                reciprocal_rank = 1.0 / rank
                hit = True
                break

        # Precision @ K
        relevant_count = sum(1 for c in chunks if _hit_in_chunk(c, expected_keywords))
        precision = relevant_count / len(chunks) if chunks else 0.0

        return {
            "question": question,
            "hit": hit,
            "reciprocal_rank": round(reciprocal_rank, 4),
            "precision": round(precision, 4),
            "latency_ms": latency_ms,
            "chunks_retrieved": len(chunks),
            "top_source": chunks[0].get("filename", "N/A") if chunks else "N/A",
        }

    def run_eval(
        self,
        eval_file: Optional[Path] = None,
        results_file: Optional[Path] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite and return aggregated metrics.

        Args:
            eval_file: Path to eval_questions.json. Uses default if None.
            results_file: Path to save eval_results.json. Uses default if None.
            verbose: If True, print per-question results to terminal.

        Returns:
            Dict with aggregate metrics and per-question results.
        """
        eval_path = eval_file or DEFAULT_EVAL_FILE
        results_path = results_file or DEFAULT_RESULTS_FILE
        questions = _load_eval_questions(eval_path)

        if verbose:
            self._print_header(len(questions))

        per_question_results = []
        for i, item in enumerate(questions, start=1):
            q = item.get("question", "")
            keywords = item.get("expected_keywords", [])
            category = item.get("category", "general")

            if not q:
                continue

            result = self.evaluate_single(q, keywords)
            result["category"] = category
            per_question_results.append(result)

            if verbose:
                self._print_row(i, result)

        # Aggregate metrics
        total = len(per_question_results)
        if total == 0:
            return {"error": "No questions evaluated."}

        hit_count = sum(1 for r in per_question_results if r.get("hit"))
        avg_mrr = sum(r.get("reciprocal_rank", 0.0) for r in per_question_results) / total
        avg_precision = sum(r.get("precision", 0.0) for r in per_question_results) / total
        avg_latency = sum(r.get("latency_ms", 0) for r in per_question_results) / total

        aggregate = {
            "total_questions": total,
            "hit_rate_at_k": round(hit_count / total, 4),
            "mrr": round(avg_mrr, 4),
            "avg_precision_at_k": round(avg_precision, 4),
            "avg_latency_ms": round(avg_latency),
            "hits": hit_count,
            "misses": total - hit_count,
        }

        if verbose:
            self._print_summary(aggregate)

        # Save results
        full_results = {"aggregate": aggregate, "per_question": per_question_results}
        try:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=2)
            if verbose:
                print(f"\n[+] Full results saved to: {results_path}")
        except Exception as exc:
            logger.warning(f"Could not save eval results: {exc}")

        return full_results

    # ── Terminal Formatting ──────────────────────────────────────────────────

    def _print_header(self, n_questions: int):
        print("\n" + "=" * 72)
        print("  BYTEFORCE-RAG  ·  RETRIEVAL ACCURACY BENCHMARK")
        print("  Mangalore Refinery & Petrochemicals Ltd (MRPL) · SIH 26117")
        print("=" * 72)
        print(f"  Questions : {n_questions}    Top-K : {self.top_k}")
        print("=" * 72)
        print(f"{'#':<4} {'HIT':<5} {'MRR':<7} {'P@K':<7} {'Lat(ms)':<9} {'Top Source'}")
        print("-" * 72)

    def _print_row(self, idx: int, result: Dict):
        hit_icon = "✓" if result.get("hit") else "✗"
        mrr = result.get("reciprocal_rank", 0.0)
        prec = result.get("precision", 0.0)
        lat = result.get("latency_ms", 0)
        src = result.get("top_source", "N/A")[:35]
        q_short = result.get("question", "")[:55]

        print(f"{idx:<4} {hit_icon:<5} {mrr:<7.4f} {prec:<7.4f} {lat:<9} {src}")
        print(f"     Q: {q_short}")
        print()

    def _print_summary(self, agg: Dict):
        print("=" * 72)
        print("  AGGREGATE RESULTS")
        print("=" * 72)
        hit_rate_pct = agg["hit_rate_at_k"] * 100
        mrr = agg["mrr"]
        prec_pct = agg["avg_precision_at_k"] * 100
        lat = agg["avg_latency_ms"]
        hits = agg["hits"]
        total = agg["total_questions"]

        print(f"  Hit Rate @ {self._get_rag().retrieve.__defaults__[0] if False else self.top_k}  :  {hit_rate_pct:.1f}%   ({hits}/{total} questions answered)")
        print(f"  MRR            :  {mrr:.4f}  (1.0 = perfect, first result always correct)")
        print(f"  Precision @ K  :  {prec_pct:.1f}%  (avg relevant chunks in top-{self.top_k})")
        print(f"  Avg Latency    :  {lat} ms per query")

        # Performance rating
        if hit_rate_pct >= 80:
            rating = "EXCELLENT ★★★★★"
        elif hit_rate_pct >= 60:
            rating = "GOOD ★★★★"
        elif hit_rate_pct >= 40:
            rating = "FAIR ★★★  — consider reingesting or expanding data"
        else:
            rating = "POOR ★★   — check data ingestion and similarity threshold"

        print(f"\n  Overall Rating :  {rating}")
        print("=" * 72 + "\n")


def run_eval(top_k: int = 5, eval_file: Optional[Path] = None) -> Dict[str, Any]:
    """Module-level convenience entry point."""
    evaluator = RAGEvaluator(top_k=top_k)
    return evaluator.run_eval(eval_file=eval_file, verbose=True)


if __name__ == "__main__":
    run_eval()
