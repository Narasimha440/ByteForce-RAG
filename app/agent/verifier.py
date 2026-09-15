import json
from typing import Dict, Any, List

class EvidenceVerifier:
    def __init__(self, llm_call):
        self.llm_call = llm_call

    def verify(
        self,
        question: str,
        context: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not context or not context.strip() or not sources:
            return {
                "sufficient": False,
                "reason": "No usable local evidence was retrieved.",
                "supported_points": [],
                "missing_points": [question],
            }

        source_summary = []

        for index, source in enumerate(sources, start=1):
            filename = source.get(
                "filename",
                source.get("source", "Unkown Document"),
            )

            location = source.get(
                "location",
                source.get("page", "Unkown Location"),
            )

            score = source.get("score")

            source_summary.append(
                {
                    "evidence_id": f"EVIDENCE_{index}",
                    "filename": filename,
                    "location": location,
                    "score": score,
                }
            )

            prompt = f"""
You are an evidence verification component inside a private,
on-premise industrial RAG system.

Your ONLY job is to determine whether the retrieved local evidence
is sufficient and relevant enough to answer the user's question.

IMPORTANT RULES:

1. Use ONLY the retrieved evidence.
2. Do NOT use outside knowledge.
3. Do NOT answer the user's question.
4. Do NOT assume that retrieved documents are relevant merely because
   they have a similarity score.
5. The evidence must directly support the requested subject and
   important parts of the question.
6. If the question asks for multiple pieces of information and the
   evidence does not support important parts, mark it insufficient.
7. If documents are about a different subject, mark the evidence
   insufficient.
8. If evidence is ambiguous or unrelated, mark it insufficient.
9. Be conservative. When uncertain, return false.
10. Never invent missing evidence.

Return ONLY valid JSON in exactly this structure:

{{
  "sufficient": true,
  "reason": "Short explanation of why the evidence is sufficient or insufficient.",
  "supported_points": [
    "Point supported by the evidence"
  ],
  "missing_points": [
    "Important requested point not supported by the evidence"
  ]
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

            if result is None:
                return {
                    "sufficient": False,
                    "reason": "Evidence verification returned an invalid result.",
                    "supported_points": [],
                    "missing_points": [question],
                }

            sufficient = result.get("sufficient", False)

            if not isinstance(sufficient, bool):
                sufficient = str(sufficient).lower() == "true"

            return {
                "sufficient": sufficient,
                "reason": str(
                    result.get(
                        "reason",
                        "No verification reason provided.",
                    )
                ),
                "supported_points": self._clean_list(
                    result.get("supported_points", [])
                ),
                "missing_points": self._clean_list(
                    result.get("missing_points", [])
                ),
            }

        except Exception as e:
            return {
                "sufficient": False,
                "reason": f"Evidence verification failed: {e}",
                "supported_points": [],
                "missing_points": [question],
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


        if "``" in raw:
            parts = raw.split("``")

            for part in parts:

                part = part.strip()

                if part.startswith("json"):
                    part = part[4:].strip()

                try:
                    data = json.loads(part)

                    if isinstance(data, dict):
                        return data

                except json.JSONDecodeError:
                    continue

        start = raw.find("{")
        end = raw.find("}")

        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(
                    raw[start : end + 1]
                )

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