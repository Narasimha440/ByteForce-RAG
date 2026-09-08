import json
import re
from typing import Any, Dict, List


class EvidenceVerifier:
    """
    Verifies whether LocalRAG.search() returned evidence that actually
    supports the user's question. Retrieval scores are treated as retrieval
    scores, not calibrated probabilities.
    """

    def __init__(self, llm_call):
        self.llm_call = llm_call

    def verify(
        self,
        question: str,
        rag_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        rag_result = rag_result or {}

        found = bool(rag_result.get("found", False))
        context = str(rag_result.get("context", "") or "")
        sources = rag_result.get("sources", []) or []
        retrieval_score = float(rag_result.get("confidence", 0.0) or 0.0)

        if not found or not context.strip() or not sources:
            return {
                "sufficient": False,
                "verified": False,
                "reason": "No usable local evidence was retrieved.",
                "supported_points": [],
                "missing_points": [question],
                "score": retrieval_score,
                "retrieval_score": retrieval_score,
            }

        if "[SAFETY ADVISORY:" in context:
            return {
                "sufficient": False,
                "verified": False,
                "reason": (
                    "The retrieved evidence does not provide verified "
                    "numerical engineering information for this request."
                ),
                "supported_points": [],
                "missing_points": [question],
                "score": retrieval_score,
                "retrieval_score": retrieval_score,
            }

        source_summary: List[Dict[str, Any]] = []

        for index, source in enumerate(sources, start=1):
            source_summary.append({
                "evidence_id": f"EVIDENCE_{index}",
                "filename": source.get(
                    "filename",
                    source.get("source", "Unknown Document"),
                ),
                "location": source.get(
                    "location",
                    f"Page {source.get('page_number', 'Unknown')}",
                ),
                "score": source.get("score"),
            })

        prompt = f"""
You are an evidence verification component inside a private,
on-premise industrial RAG system.

Your ONLY job is to determine whether the retrieved local evidence
is sufficient and relevant enough to answer the user's question.

Rules:
1. Use ONLY the retrieved evidence.
2. Do NOT use outside knowledge.
3. Do NOT answer the user's question.
4. Do not assume relevance merely because a retrieval score exists.
5. The evidence must directly support the subject and important parts
   of the question.
6. If multiple pieces of information are requested and important parts
   are missing, mark insufficient.
7. If the documents are about a different subject, mark insufficient.
8. Be conservative. When uncertain, return false.
9. Never invent missing evidence.

Return ONLY valid JSON:
{{
  "sufficient": true,
  "reason": "Short explanation",
  "supported_points": [],
  "missing_points": []
}}

USER QUESTION:
{question}

RETRIEVED SOURCES:
{json.dumps(source_summary, indent=2)}

RETRIEVED LOCAL EVIDENCE:
{context}
"""

        try:
            raw = self.llm_call(prompt, temperature=0.0)
            result = self._parse_json(raw)

            if not result:
                return self._fallback(
                    retrieval_score,
                    "Evidence verifier returned an invalid result.",
                    question,
                )

            sufficient = result.get("sufficient", False)
            if not isinstance(sufficient, bool):
                sufficient = str(sufficient).lower() == "true"

            return {
                "sufficient": sufficient,
                "verified": sufficient,
                "reason": str(
                    result.get(
                        "reason",
                        "Evidence verification completed.",
                    )
                ),
                "supported_points": self._clean_list(
                    result.get("supported_points", [])
                ),
                "missing_points": self._clean_list(
                    result.get("missing_points", [])
                ),
                "score": retrieval_score,
                "retrieval_score": retrieval_score,
            }

        except Exception as exc:
            # Conservative but usable fallback: a healthy retrieval with
            # a reasonable score can proceed; otherwise force re-planning.
            sufficient = retrieval_score >= 0.55

            return {
                "sufficient": sufficient,
                "verified": sufficient,
                "reason": (
                    "Verifier LLM unavailable; "
                    + (
                        "retrieval score passed fallback threshold."
                        if sufficient
                        else "retrieval evidence was too weak."
                    )
                ),
                "supported_points": [],
                "missing_points": [] if sufficient else [question],
                "score": retrieval_score,
                "retrieval_score": retrieval_score,
                "verifier_error": str(exc),
            }

    @staticmethod
    def _fallback(
        score: float,
        reason: str,
        question: str,
    ) -> Dict[str, Any]:
        sufficient = score >= 0.55
        return {
            "sufficient": sufficient,
            "verified": sufficient,
            "reason": reason,
            "supported_points": [],
            "missing_points": [] if sufficient else [question],
            "score": score,
            "retrieval_score": score,
        }

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any] | None:
        if not raw:
            return None

        raw = raw.strip()

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _clean_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
