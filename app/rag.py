"""
Local Retrieval-Augmented Generation (RAG) Engine for SIH 26117.

Features:
- Sovereign, 100% air-gapped retrieval using Hybrid Search (Dense BGE-M3 + Lexical BM25)
- Reciprocal Rank Fusion (RRF) combining semantic intent with exact equipment tag matching
- Local Reranker (Cross-Encoder / Lexical Density) for maximum precision
- Strict similarity threshold filtering to prevent hallucination and noise
- SIH-compliant evidence formatting with full document, page, and extraction provenance
- Exposes clean RAG APIs: retrieve(), build_context(), and answer()
- Safe fallback when knowledge is insufficient:
  "The local knowledge base does not contain enough relevant information to answer this."
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from .bm25 import BM25Index
from .config import (
    BM25_INDEX_PATH,
    HYBRID_RETRIEVAL_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RERANKER_ENGINE,
    RERANKER_MODEL,
    RERANKING_ENABLED,
    RRF_K,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from .domain_expansion import expand_refinery_query, extract_query_equipment_tags
from .embeddings import get_embedding_provider
from .equipment_graph import format_dossier_box, get_equipment_dossier
from .reranker import get_reranker
from .store import VectorStore

logger = logging.getLogger(__name__)

INSUFFICIENT_KNOWLEDGE_MESSAGE = (
    "The local knowledge base does not contain enough relevant information to answer this."
)


def verify_numerical_grounding(question: str, hits: List[Any]) -> Dict[str, Any]:
    """
    Checks whether a query requesting quantitative/safety engineering parameters
    (pressure, temperature, stroke time, limit, thickness, sulfur, voltage)
    is backed by verified numbers and engineering units in the retrieved evidence.
    """
    if not question:
        return {"grounded": True, "details": "No quantitative constraints detected."}

    q_lower = question.lower()
    numeric_keywords = [
        "pressure", "temperature", "temp", "limit", "stroke", "closing time",
        "thickness", "sulfur", "ppm", "bar", "°c", "celsius", "kv", "vdc", "flow", "set point"
    ]
    is_numeric_query = any(k in q_lower for k in numeric_keywords)

    if not is_numeric_query:
        return {"grounded": True, "details": "General qualitative query."}

    # Search for engineering units and digits in evidence text
    unit_regex = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(?:bar|bar\(g\)|°c|c|ppm|seconds|sec|s|mm|m3/h|vdc|vac|kpa|%)\b",
        re.IGNORECASE,
    )

    found_measurements = []
    for hit in hits:
        payload = hit.payload if hasattr(hit, "payload") and hit.payload else (hit if isinstance(hit, dict) else {})
        text = payload.get("text", "")
        matches = unit_regex.findall(text)
        if matches:
            found_measurements.extend(matches)

    if found_measurements:
        return {
            "grounded": True,
            "measurements_found": len(found_measurements),
            "details": f"Verified {len(found_measurements)} quantitative engineering measurement(s) in local evidence.",
        }
    else:
        return {
            "grounded": False,
            "measurements_found": 0,
            "details": "Quantitative safety parameters requested, but evidence contains no verified numerical values.",
        }


class LocalRAG:

    def __init__(self, embedding_provider=None, vector_store=None, bm25_index=None, reranker=None):
        logger.info("Initializing LocalRAG...")
        self.embedder = embedding_provider or get_embedding_provider()
        vector_size = self.embedder.get_dimension()
        self.store = vector_store or VectorStore(vector_size=vector_size)
        self.store.ensure_collection()

        # BM25 Lexical Engine
        self.bm25 = bm25_index
        if self.bm25 is None and BM25_INDEX_PATH.exists():
            try:
                self.bm25 = BM25Index()
                self.bm25.load(BM25_INDEX_PATH)
                logger.info(f"Loaded BM25 index with {self.bm25.corpus_size} chunks.")
            except Exception as exc:
                logger.warning(f"Could not load BM25 index: {exc}")
                self.bm25 = None

        # Reranker Engine
        self.reranker = reranker
        if self.reranker is None and RERANKING_ENABLED:
            self.reranker = get_reranker(RERANKER_ENGINE, RERANKER_MODEL)

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using Hybrid Search (Dense + BM25) and RRF.

        Returns:
            List of candidate dictionaries passing threshold and reranking.
        """
        if not question or not question.strip():
            return []

        clean_query = question.strip()

        # Domain Query Expansion (Offline MRPL Ontology)
        augmented_query, lexical_boosts = expand_refinery_query(clean_query)

        # 1. Dense Semantic Retrieval via Qdrant
        query_vector = self.embedder.embed_query(augmented_query)
        dense_hits = self.store.search(
            query_vector,
            limit=top_k * 4 if HYBRID_RETRIEVAL_ENABLED else top_k,
            filter_dict=filter_criteria,
        )

        dense_candidates = []
        for rank, hit in enumerate(dense_hits):
            payload = dict(hit.payload or {})
            payload["score"] = float(hit.score) if hit.score is not None else 0.0
            dense_candidates.append((payload, rank))

        # 2. Sparse Lexical Retrieval via BM25 with Lexical Boosts
        sparse_candidates = []
        if HYBRID_RETRIEVAL_ENABLED and self.bm25 and self.bm25.corpus_size > 0:
            bm25_query = clean_query
            if lexical_boosts:
                bm25_query += " " + " ".join(lexical_boosts[:4])

            bm25_results = self.bm25.search(bm25_query, top_k=top_k * 4)
            for rank, (payload, b_score) in enumerate(bm25_results):
                p_copy = dict(payload)
                p_copy["bm25_score"] = float(b_score)
                sparse_candidates.append((p_copy, rank))

        # 3. Reciprocal Rank Fusion (RRF)
        if sparse_candidates:
            rrf_scores: Dict[str, float] = {}
            candidate_map: Dict[str, Dict[str, Any]] = {}

            # Score dense candidates
            for payload, rank in dense_candidates:
                doc_id = payload.get("document_id", "")
                chunk_id = payload.get("chunk_id", str(rank))
                unique_key = f"{doc_id}:{chunk_id}"
                rrf_scores[unique_key] = rrf_scores.get(unique_key, 0.0) + (1.0 / (RRF_K + rank))
                candidate_map[unique_key] = payload

            # Score sparse candidates
            for payload, rank in sparse_candidates:
                doc_id = payload.get("document_id", "")
                chunk_id = payload.get("chunk_id", str(rank))
                unique_key = f"{doc_id}:{chunk_id}"
                rrf_scores[unique_key] = rrf_scores.get(unique_key, 0.0) + (1.0 / (RRF_K + rank))
                if unique_key not in candidate_map:
                    candidate_map[unique_key] = payload
                else:
                    # Enrich candidate with sparse score
                    candidate_map[unique_key]["bm25_score"] = payload.get("bm25_score", 0.0)

            fused_candidates = []
            for u_key, rrf_score in rrf_scores.items():
                cand = candidate_map[u_key]
                cand["rrf_score"] = rrf_score
                dense_score = float(cand.get("score", 0.0) or 0.0)
                bm25_score = float(cand.get("bm25_score", 0.0) or 0.0)

                # Calibrate unified similarity score
                if dense_score > 0.0 and bm25_score > 0.0:
                    unified_score = 0.5 * dense_score + 0.5 * min(0.95, bm25_score / 12.0)
                elif dense_score > 0.0:
                    unified_score = dense_score
                else:
                    unified_score = min(0.90, bm25_score / 12.0)

                cand["score"] = round(unified_score, 4)
                fused_candidates.append(cand)

            fused_candidates.sort(key=lambda x: (x.get("rrf_score", 0.0), x.get("score", 0.0)), reverse=True)
            candidates = fused_candidates[:top_k * 3]
        else:
            candidates = [p for p, _ in dense_candidates]

        # 4. Filter by minimum similarity threshold
        filtered_candidates = [
            c for c in candidates
            if c.get("score", 0.0) >= threshold
        ]

        # 4.5. P1 FIX: Page Diversity Filter
        # Ensures we don't return 5 chunks from the exact same page unless necessary
        diverse_candidates = []
        page_counts = {}
        for c in filtered_candidates:
            doc_id = c.get("filename", "unknown")
            page_num = c.get("page_number", "unknown")
            key = f"{doc_id}:{page_num}"
            count = page_counts.get(key, 0)
            if count < 2:  # Max 2 chunks per page
                diverse_candidates.append(c)
                page_counts[key] = count + 1

        # 5. Rerank top candidates if enabled
        if RERANKING_ENABLED and self.reranker and diverse_candidates:
            final_hits = self.reranker.rerank(clean_query, diverse_candidates, top_k=top_k)
        else:
            final_hits = diverse_candidates[:top_k]

        return final_hits

    def build_context(
        self,
        hits: List[Any],
        expand_to_parent: bool = True,
        question: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build clear, SIH-traceable evidence blocks, equipment dossiers, and structured source citations.
        Supports Hierarchical Parent-Child Context Expansion to prevent procedural fragmentation.

        Returns:
            Tuple of (formatted_context_string, structured_sources_list)
        """
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        seen_parent_ids: Set[str] = set()

        # Check for Equipment Tags in Question and Prepend Dossier
        if question:
            detected_tags = extract_query_equipment_tags(question)
            for tag in detected_tags:
                dossier = get_equipment_dossier(tag, hits)
                if dossier:
                    context_parts.append(format_dossier_box(dossier))

            # Numerical Grounding Check
            grounding = verify_numerical_grounding(question, hits)
            if not grounding["grounded"]:
                context_parts.append(
                    "[SAFETY ADVISORY: Quantitative engineering parameters requested, but retrieved "
                    "evidence contains no verified numerical limits. Verify with shift supervisor before field action.]\n"
                )

        for index, hit in enumerate(hits, start=1):
            payload = hit.payload if hasattr(hit, "payload") and hit.payload else (hit if isinstance(hit, dict) else {})

            filename = payload.get("filename", "Unknown")
            source = payload.get("source", filename)
            location = payload.get("location", "Unknown")
            page_number = payload.get("page_number", "N/A")
            section = payload.get("section") or "General"
            category = payload.get("category", "general")
            document_type = payload.get("document_type", "general")
            extraction = payload.get("extraction_method", "native_text")
            ocr_used = payload.get("ocr_used", False)
            tags = payload.get("tags", [])
            chunk_id = payload.get("chunk_id", "N/A")
            parent_id = payload.get("parent_id")
            parent_text = payload.get("parent_text")
            score = payload.get("score", getattr(hit, "score", 0.0))
            is_table = payload.get("is_table", False)

            # Hierarchical Context Expansion: use parent_text for procedure continuity if available
            if expand_to_parent and parent_text and parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                content_label = "Complete Section Context" if not is_table else "Complete Table Context"
                text = parent_text.strip()
            else:
                content_label = "Precise Chunk Content"
                text = payload.get("text", "").strip()

            evidence_block = f"""[EVIDENCE {index}] [E{index}]
Document: {filename}
Page: {page_number}
Location: {location}
Section: {section}
Category: {category}
Document Type: {document_type}
Extraction Method: {extraction} (OCR Used: {ocr_used})
Industrial Tags: {", ".join(tags) if tags else "None"}
Relevance: {score}

Text:
{text}
"""
            context_parts.append(evidence_block)

            # Visual / Multimodal Handoff for Bhuvan's Vision Pipeline (Qwen2.5-VL-7B)
            is_visual = (
                payload.get("requires_vision_agent", False)
                or payload.get("content_type") in ("cad_drawing", "scanned_document", "image")
                or payload.get("file_type") in ("dwg", "dxf", "png", "jpg", "jpeg", "webp")
                or ocr_used
            )
            visual_handoff = None
            if is_visual:
                visual_handoff = {
                    "requires_vision_agent": True,
                    "target_model": "Qwen2.5-VL-7B-Instruct",
                    "document": filename,
                    "source_path": source,
                    "page_number": page_number,
                    "file_type": payload.get("file_type", "unknown"),
                    "drawing_code": payload.get("drawing_code"),
                    "equipment_tags": tags,
                    "recommended_action": (
                        "Inspect engineering schematic visually for piping connectivity, signal lines, and layout"
                        if payload.get("file_type") in ("dwg", "dxf")
                        else "Analyze rendered high-res image for visual details"
                    ),
                }

            source_item = {
                "filename": filename,
                "source": source,
                "location": location,
                "page_number": page_number,
                "section": section,
                "category": category,
                "document_type": document_type,
                "extraction_method": extraction,
                "ocr_used": ocr_used,
                "tags": tags,
                "chunk_id": chunk_id,
                "score": round(float(score), 4) if score is not None else 0.0,
            }
            if visual_handoff:
                source_item["visual_handoff"] = visual_handoff

            sources.append(source_item)

        return "\n".join(context_parts), sources

    def answer(
        self,
        question: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Answer question using local knowledge base and local Ollama LLM.
        Never hallucinates if retrieval confidence is insufficient.
        """
        hits = self.retrieve(
            question,
            top_k=top_k,
            threshold=threshold,
            filter_criteria=filter_criteria,
        )

        if not hits:
            return INSUFFICIENT_KNOWLEDGE_MESSAGE, []

        context, sources = self.build_context(hits, question=question)

        prompt = f"""You are a private industrial knowledge assistant for sovereign confidential industrial facilities.

Answer the user's question using ONLY the evidence provided from the LOCAL KNOWLEDGE BASE below.

Rules:
1. Do not invent requirements, standards, procedures, measurements, equipment specifications, safety rules, or engineering values.
2. If the evidence is insufficient to answer the question with certainty, respond exactly with:
   "{INSUFFICIENT_KNOWLEDGE_MESSAGE}"
3. Prefer information directly supported by the evidence.
4. Mention the relevant source document and page number for each key statement.

USER QUESTION:
{question}

LOCAL KNOWLEDGE BASE EVIDENCE:
{context}

Answer concisely, accurately, and professionally:
"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Answer only from the provided local evidence. Never hallucinate.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            answer_text = data["message"]["content"]
            return answer_text, sources

        except Exception as err:
            logger.warning(f"Ollama call failed ({err}). Returning structured evidence directly.")
            evidence_summary = (
                f"Retrieved {len(sources)} relevant evidence chunk(s) from the local knowledge base, "
                f"but the local LLM ({OLLAMA_MODEL}) was unreachable.\n\n"
                f"Primary source: {sources[0]['filename']} ({sources[0]['location']})\n\n"
                f"Evidence text:\n{sources[0].get('filename')}:\n"
                f"{hits[0].get('text', '') if isinstance(hits[0], dict) else getattr(hits[0], 'payload', {}).get('text', '')}"
            )
            return evidence_summary, sources

    def search(self, question: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> dict:
        """
        Agent Interface Contract (P1).
        Returns a strict JSON-compatible dictionary for downstream AI Agents.
        """
        hits = self.retrieve(question, top_k=top_k, threshold=threshold)
        
        if not hits:
            return {
                "found": False,
                "confidence": 0.0,
                "context": "",
                "sources": [],
                "reason": "insufficient_local_evidence"
            }
            
        context, sources = self.build_context(hits, question=question)
        
        # Calculate an aggregate confidence based on top scores
        top_score = sources[0]["score"] if sources else 0.0
        visual_handoffs = get_visual_evidence(hits)

        return {
            "found": True,
            "confidence": top_score,
            "context": context,
            "sources": sources,
            "visual_handoffs": visual_handoffs,
            "retrieval_stats": {
                "candidates": top_k * 4 if HYBRID_RETRIEVAL_ENABLED else top_k,
                "reranked": len(sources),
                "final": min(len(sources), top_k)
            }
        }

    def close(self):
        """Close vector store connection cleanly."""
        if hasattr(self.store, "close"):
            self.store.close()


def get_visual_evidence(hits: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract visual and CAD handoff objects from retrieval hits
    for downstream consumption by Bhuvan's Vision Agent (Qwen2.5-VL).
    """
    visuals: List[Dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload if hasattr(hit, "payload") and hit.payload else (hit if isinstance(hit, dict) else {})
        if not payload:
            continue

        ocr_used = payload.get("ocr_used", False)
        content_type = payload.get("content_type", "")
        file_type = payload.get("file_type", "")

        if (
            payload.get("requires_vision_agent")
            or content_type in ("cad_drawing", "scanned_document", "image")
            or file_type in ("dwg", "dxf", "png", "jpg", "jpeg", "webp")
            or ocr_used
        ):
            visuals.append({
                "document": payload.get("filename", "unknown"),
                "source_path": payload.get("source", ""),
                "page_number": payload.get("page_number", 1),
                "file_type": file_type,
                "drawing_code": payload.get("drawing_code"),
                "equipment_tags": payload.get("tags", []),
                "target_model": "Qwen2.5-VL-7B-Instruct",
                "requires_vision_agent": True,
            })
    return visuals


def retrieve(question: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> List[Any]:
    """Clean module-level API for retrieval."""
    rag = LocalRAG()
    return rag.retrieve(question, top_k=top_k, threshold=threshold)


def build_context(
    hits: List[Any],
    expand_to_parent: bool = True,
    question: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Clean module-level API for context construction."""
    rag = LocalRAG()
    return rag.build_context(hits, expand_to_parent=expand_to_parent, question=question)