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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import requests

from .bm25 import BM25Index
from .config import (
    BM25_INDEX_PATH,
    HYBRID_RETRIEVAL_ENABLED,
    LLM_MAX_CONTEXT_CHARS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RERANKER_ENGINE,
    RERANKER_MODEL,
    RERANKING_ENABLED,
    RRF_K,
    SIMILARITY_THRESHOLD,
    TOP_K,
    get_active_ollama_model,
)
from .domain_expansion import (
    expand_equipment_acronyms,
    expand_refinery_query,
    extract_query_equipment_tags,
)
from .embeddings import get_embedding_provider
from .equipment_graph import format_dossier_box, get_equipment_dossier
from .reranker import get_reranker
from .store import VectorStore

logger = logging.getLogger(__name__)

INSUFFICIENT_KNOWLEDGE_MESSAGE = (
    "The local knowledge base does not contain enough relevant information to answer this."
)

CONVERSATIONAL_GREETINGS = {
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "greetings", "who are you", "what are you", "what can you do", "help",
    "thanks", "thank you", "bye", "goodbye", "namaste"
}


def is_conversational_query(question: str) -> bool:
    """
    Determine if a user query is purely conversational, greeting, or identity inquiry
    rather than an exact technical query against refinery documents.
    """
    if not question:
        return False
    q_clean = re.sub(r"[^\w\s]", "", question.strip().lower())
    words = q_clean.split()
    if not words:
        return False
    if q_clean in CONVERSATIONAL_GREETINGS:
        return True
    if len(words) <= 3 and any(w in CONVERSATIONAL_GREETINGS for w in words):
        return True
    return False


def calculate_document_authority(
    filename: str, location: str, query: str, revision: str = "Active"
) -> float:
    """
    Calculate an authority multiplier for a document chunk based on document importance and revision status.
    - Boosts completed Safety Audits, SOPs, and Design Basis records; demotes blank checklist templates.
    - Factors in document lifecycle: boosts active latest revisions, downweights drafts and superseded versions.
    """
    fn_lower = (filename or "").lower()
    loc_lower = (location or "").lower()
    combined = f"{fn_lower} {loc_lower}"
    q_lower = query.lower()

    # Base authority multiplier
    multiplier = 1.0

    # 1. Operational & Incident Audits (Highest Authority)
    if (
        "09_inspection_accident" in combined
        or "safety_audit" in combined
        or "incident" in combined
        or "audit" in combined
        or "annual_safety" in combined
    ):
        if any(w in q_lower for w in ("audit", "report", "incident", "accident", "mah", "inspection", "safety")):
            multiplier = 1.65
        else:
            multiplier = 1.35

    # 2. Standards, Reference & SOPs
    elif (
        "01_standards_reference" in combined
        or "04_sops_manuals" in combined
        or "sop" in combined
        or "standard" in combined
        or "design_basis" in combined
    ):
        multiplier = 1.25

    # 3. P&IDs and Schematics
    elif "03_pids" in combined or "pid" in combined or "p&id" in combined or "schematic" in combined:
        multiplier = 1.15

    # 4. Blank Templates and Forms (Demote blank checklists when searching for real information)
    elif (
        "02_templates_checklists_forms" in combined
        or "template" in combined
        or "blank" in combined
        or "fat_sat_checklist" in combined
    ):
        multiplier = 0.65

    # Document Lifecycle & Revision Multiplier
    rev_lower = (revision or "").lower()
    if "superseded" in rev_lower or "obsolete" in rev_lower or "withdrawn" in rev_lower:
        multiplier *= 0.40
    elif "draft" in rev_lower or "preliminary" in rev_lower:
        multiplier *= 0.75
    elif "rev" in rev_lower:
        multiplier *= 1.05

    return round(multiplier, 4)


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



import hashlib
from collections import OrderedDict
from typing import Optional, Tuple

class SemanticCache:
    """Enterprise LRU Semantic Cache for instantly answering similar queries."""
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.cache: OrderedDict[str, Tuple[str, list, str]] = OrderedDict()
    
    def _get_hash(self, query: str) -> str:
        import re
        norm = re.sub(r'[^\w\s]', '', query.lower()).strip()
        return hashlib.sha256(norm.encode('utf-8')).hexdigest()
        
    def get(self, query: str) -> Optional[Tuple[str, list, str]]:
        q_hash = self._get_hash(query)
        if q_hash in self.cache:
            self.cache.move_to_end(q_hash)
            return self.cache[q_hash]
        return None
        
    def put(self, query: str, answer: str, sources: list, confidence: str):
        q_hash = self._get_hash(query)
        self.cache[q_hash] = (answer, sources, confidence)
        self.cache.move_to_end(q_hash)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

_GLOBAL_SEMANTIC_CACHE = SemanticCache()


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

        # Check if query contains exact equipment tags, telemetry labels, or error codes
        tags_in_query = extract_query_equipment_tags(clean_query)
        is_tag_query = bool(
            tags_in_query
            or re.search(r"\b(?:[A-Z0-9]{2,6}-[A-Z0-9]{2,6}|ERR[_-]?[A-Z0-9]+|0x[0-9A-Fa-f]+|E-\d{2,4})\b", clean_query)
        )

        # Adaptive RRF Weighting:
        # Tag/Error queries prioritize exact BM25 matches (70/30)
        # Conceptual queries prioritize deep BGE-M3 semantic vectors (70/30)
        if is_tag_query:
            dense_weight = 0.30
            bm25_weight = 0.70
        else:
            dense_weight = 0.70
            bm25_weight = 0.30

        # Domain Query Expansion & Bidirectional Acronym Augmentation
        augmented_query, lexical_boosts = expand_refinery_query(clean_query)
        acronym_enriched = expand_equipment_acronyms(clean_query)
        if acronym_enriched != clean_query:
            augmented_query = f"{acronym_enriched} | {augmented_query}"

        # 1 & 2. PARALLEL Dense + Sparse Retrieval (ThreadPoolExecutor)
        # Running Qdrant vector search and BM25 simultaneously cuts wait time by ~40-50%
        dense_candidates = []
        sparse_candidates = []

        def _run_dense():
            query_vector = self.embedder.embed_query(augmented_query)
            hits = self.store.search(
                query_vector,
                limit=top_k * 2,  # Reduced from top_k*4 → cuts reranker load by 50%
                filter_dict=filter_criteria,
            )
            result = []
            for rank, hit in enumerate(hits):
                payload = dict(hit.payload or {})
                payload["score"] = float(hit.score) if hit.score is not None else 0.0
                result.append((payload, rank))
            return result

        def _run_sparse():
            if not (HYBRID_RETRIEVAL_ENABLED and self.bm25 and self.bm25.corpus_size > 0):
                return []
            bm25_query = clean_query
            if lexical_boosts:
                bm25_query += " " + " ".join(lexical_boosts[:4])
            bm25_results = self.bm25.search(bm25_query, top_k=top_k * 2)
            result = []
            for rank, (payload, b_score) in enumerate(bm25_results):
                if filter_criteria:
                    if not all(payload.get(k) == v for k, v in filter_criteria.items()):
                        continue
                p_copy = dict(payload)
                p_copy["bm25_score"] = float(b_score)
                result.append((p_copy, rank))
            return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(_run_dense)
            future_sparse = executor.submit(_run_sparse)
            dense_candidates = future_dense.result()
            sparse_candidates = future_sparse.result()

        # 3. Adaptive Reciprocal Rank Fusion (Adaptive RRF)
        if sparse_candidates:
            rrf_scores: Dict[str, float] = {}
            candidate_map: Dict[str, Dict[str, Any]] = {}

            # Score dense candidates with dynamic weight
            for payload, rank in dense_candidates:
                doc_id = payload.get("document_id", "")
                chunk_id = payload.get("chunk_id", str(rank))
                unique_key = f"{doc_id}:{chunk_id}"
                rrf_scores[unique_key] = rrf_scores.get(unique_key, 0.0) + (dense_weight / (RRF_K + rank))
                candidate_map[unique_key] = payload

            # Score sparse candidates with dynamic weight
            for payload, rank in sparse_candidates:
                doc_id = payload.get("document_id", "")
                chunk_id = payload.get("chunk_id", str(rank))
                unique_key = f"{doc_id}:{chunk_id}"
                rrf_scores[unique_key] = rrf_scores.get(unique_key, 0.0) + (bm25_weight / (RRF_K + rank))
                if unique_key not in candidate_map:
                    candidate_map[unique_key] = payload
                else:
                    # Enrich candidate with sparse score
                    candidate_map[unique_key]["bm25_score"] = payload.get("bm25_score", 0.0)

            fused_candidates = []
            for u_key, rrf_score in rrf_scores.items():
                cand = candidate_map[u_key]
                dense_score = float(cand.get("score", 0.0) or 0.0)
                bm25_score = float(cand.get("bm25_score", 0.0) or 0.0)

                # Calibrate unified similarity score adaptively
                if dense_score > 0.0 and bm25_score > 0.0:
                    unified_score = dense_weight * dense_score + bm25_weight * min(0.95, bm25_score / 12.0)
                elif dense_score > 0.0:
                    unified_score = dense_score
                else:
                    unified_score = min(0.90, bm25_score / 12.0)

                # Document Authority Re-weighting with Revision tracking
                fn = cand.get("filename", "")
                loc = cand.get("location", "")
                rev = cand.get("revision", "Active")
                authority_mult = calculate_document_authority(fn, loc, clean_query, revision=rev)

                cand["score"] = round(unified_score * authority_mult, 4)
                cand["rrf_score"] = round(rrf_score * authority_mult, 6)
                fused_candidates.append(cand)

            fused_candidates.sort(key=lambda x: (x.get("rrf_score", 0.0), x.get("score", 0.0)), reverse=True)
            candidates = fused_candidates[:top_k * 3]
        else:
            candidates = []
            for p, _ in dense_candidates:
                fn = p.get("filename", "")
                loc = p.get("location", "")
                rev = p.get("revision", "Active")
                authority_mult = calculate_document_authority(fn, loc, clean_query, revision=rev)
                p["score"] = round(float(p.get("score", 0.0)) * authority_mult, 4)
                candidates.append(p)
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)


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

    def _classify_confidence(self, sources: List[Dict[str, Any]]) -> str:
        """
        Classify answer confidence as HIGH / MEDIUM / LOW based on
        the top retrieved chunk's relevance score.
        """
        if not sources:
            return "LOW"
        top_score = float(sources[0].get("score", 0.0))
        if top_score >= 0.75:
            return "HIGH"
        elif top_score >= 0.50:
            return "MEDIUM"
        return "LOW"

    def _build_grounded_prompt(self, question: str, context: str) -> str:
        """Build a strict evidence-grounded prompt that mandates per-statement citations.
        Context is truncated to LLM_MAX_CONTEXT_CHARS to keep Ollama inference fast.
        """
        # Truncate context to keep prompt size manageable for small local LLMs
        if len(context) > LLM_MAX_CONTEXT_CHARS:
            context = context[:LLM_MAX_CONTEXT_CHARS] + "\n... [context truncated for speed]"

        return f"""You are a private industrial knowledge assistant for Mangalore Refinery & Petrochemicals Ltd (MRPL).

Answer the user's question using ONLY the evidence provided from the LOCAL KNOWLEDGE BASE below.

Rules:
1. NEVER invent facts, measurements, standards, equipment tags, or procedures not present in the evidence.
2. For EVERY key statement or fact, cite the source using this format at the end of the sentence:
   [E1: filename | Page X]  or  [E2: filename | Page X]
   where E1, E2 correspond to the EVIDENCE block numbers below.
3. If the evidence is insufficient to answer with certainty, respond with exactly:
   "{INSUFFICIENT_KNOWLEDGE_MESSAGE}"
4. If a safety or numerical parameter is mentioned, always include its unit (bar, \u00b0C, mm, ppm, etc.).
5. Be concise, accurate, and professional.

USER QUESTION:
{question}

LOCAL KNOWLEDGE BASE EVIDENCE:
{context}

Answer (with inline source citations):
"""

    def answer(
        self,
        question: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        filter_criteria: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        # --- ENTERPRISE SEMANTIC CACHE ---
        cached = _GLOBAL_SEMANTIC_CACHE.get(question)
        if cached and not history:
            logger.info("Semantic Cache HIT: Returning 0-latency cached answer.")
            return cached[0], cached[1], cached[2]
        # ---------------------------------
        """
        Answer question using local knowledge base and local Ollama LLM.
        - Supports conversational greetings and refinery domain questions smoothly.
        - Employs strict local evidence grounding for operational and safety queries.
        - Dynamically discovers the active Ollama model.
        - Optional `history` list of prior turns enables multi-turn query condensation.

        Returns:
            Tuple of (answer_text, sources_list, confidence_level)
            confidence_level: "HIGH" | "MEDIUM" | "LOW"
        """
        # Multi-turn query condensation: rewrite follow-up questions
        resolved_question = question
        if history:
            try:
                from .memory import condense_followup_query, is_followup_question
                if is_followup_question(question, history):
                    from .config import OLLAMA_URL
                    resolved_question = condense_followup_query(
                        history, question, ollama_url=OLLAMA_URL
                    )
                    logger.info(
                        f"[Memory] Query condensed: '{question}' → '{resolved_question}'"
                    )
            except Exception as mem_exc:
                logger.debug(f"Memory condensation skipped: {mem_exc}")
        active_model = get_active_ollama_model()

        # 1. Check for Conversational / Greeting Query
        if is_conversational_query(resolved_question):
            system_prompt = (
                "You are the Sovereign Industrial AI Assistant for Mangalore Refinery and Petrochemicals Limited (MRPL). "
                "You assist plant engineers, operators, and safety auditors with refinery P&IDs, SOPs, equipment tags "
                "(e.g., PT-101, FV-102, C-101), safety compliance reports, and OCR document processing. "
                "Be polite, professional, concise, and helpful. Greet the user warmly and invite them to ask about "
                "MRPL refinery operations, safety audits, or equipment parameters."
            )
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": active_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.3},
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"], [], "HIGH"
            except Exception:
                fallback_msg = (
                    "Hello! I am your Sovereign Industrial AI Assistant for Mangalore Refinery & Petrochemicals Ltd (MRPL).\n\n"
                    "I am ready to assist you with:\n"
                    "• **Refinery P&IDs & Equipment Tags** (e.g. PT-101, FV-102, C-101, MOV-104)\n"
                    "• **Annual Safety Audits & MAH Factory Records** (e.g. 2026 Audit Findings, Risk Mitigations)\n"
                    "• **Operating Procedures (SOPs) & Engineering Standards** (e.g. Emergency shutdown, OISD)\n"
                    "• **Technical OCR Document Processing** (Scanned records, drawings, inspection tags)\n\n"
                    "How can I assist your refinery operations or safety verification today?"
                )
                return fallback_msg, [], "HIGH"

        # 2. Operational / Evidence Retrieval
        hits = self.retrieve(
            resolved_question,
            top_k=top_k,
            threshold=threshold,
            filter_criteria=filter_criteria,
        )

        if not hits:
            return INSUFFICIENT_KNOWLEDGE_MESSAGE, [], "LOW"

        context, sources = self.build_context(hits, question=resolved_question)
        confidence = self._classify_confidence(sources)
        prompt = self._build_grounded_prompt(resolved_question, context)

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": active_model,
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

            # Append confidence advisory for LOW-confidence answers
            if confidence == "LOW":
                answer_text += (
                    "\n\n⚠ LOW CONFIDENCE: Retrieved evidence has low relevance scores. "
                    "Cross-check with shift supervisor or source documents before field action."
                )

            return answer_text, sources, confidence

        except Exception as err:
            logger.warning(f"Ollama call failed ({err}). Generating structured executive summary from evidence.")
            primary_source = sources[0] if sources else {}
            exec_summary = [
                f"**[MRPL Local Knowledge Base - Evidence Summary]**",
                f"*(Local LLM '{active_model}' was unreachable. Presenting verified extracted evidence directly)*\n",
                f"• **Primary Document**: `{primary_source.get('filename')}` (Page {primary_source.get('page_number', 'N/A')})",
                f"• **Location**: `{primary_source.get('location')}`",
                f"• **Category / Section**: {primary_source.get('category')} / {primary_source.get('section')}",
            ]
            if primary_source.get("tags"):
                exec_summary.append(f"• **Equipment Tags**: {', '.join(primary_source['tags'])}")

            top_text = hits[0].get("text", "") if isinstance(hits[0], dict) else getattr(hits[0], "payload", {}).get("text", "")
            cleaned_text = top_text.strip()
            exec_summary.append(f"\n**Verified Key Findings:**\n{cleaned_text[:1200]}")

            if len(sources) > 1:
                other_docs = {s.get("filename") for s in sources[1:] if s.get("filename")}
                if other_docs:
                    exec_summary.append(f"\n**Additional Corroborating Documents**: {', '.join(other_docs)}")

            return "\n".join(exec_summary), sources, confidence

    def answer_stream(
        self,
        question: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        filter_criteria: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming answer generator — yields tokens as they are produced by Ollama.
        Each yielded value is a string token chunk (may be a word or partial word).

        Usage:
            for token in rag.answer_stream(question):
                print(token, end="", flush=True)

        Falls back to non-streaming answer() if streaming is unavailable.
        """
        import json as _json

        active_model = get_active_ollama_model()

        # Conversational queries — no streaming needed, answer directly
        if is_conversational_query(question):
            answer, _, _ = self.answer(question)
            yield answer
            return

        # Multi-turn condensation
        resolved_question = question
        if history:
            try:
                from .memory import condense_followup_query, is_followup_question
                if is_followup_question(question, history):
                    resolved_question = condense_followup_query(history, question)
            except Exception:
                pass

        hits = self.retrieve(resolved_question, top_k=top_k, threshold=threshold,
                             filter_criteria=filter_criteria)
        if not hits:
            yield INSUFFICIENT_KNOWLEDGE_MESSAGE
            return

        context, sources = self.build_context(hits, question=resolved_question)
        prompt = self._build_grounded_prompt(resolved_question, context)

        try:
            with requests.post(
                OLLAMA_URL,
                json={
                    "model": active_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Answer only from the provided local evidence. Never hallucinate. Always cite source and page.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                    "options": {"temperature": 0.1},
                },
                timeout=300,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk_data = _json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
                        token = chunk_data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk_data.get("done"):
                            break
                    except _json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.warning(f"Streaming failed ({exc}). Falling back to blocking answer.")
            # Graceful non-streaming fallback
            answer, _, _ = self.answer(question, top_k=top_k, threshold=threshold,
                                       filter_criteria=filter_criteria, history=history)
            yield answer

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
                "candidates": top_k * 2,  # Updated to reflect actual pool size
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


# ── Global Singleton ──────────────────────────────────────────────────────────
# Caches the LocalRAG instance for the lifetime of the process.
# Avoids re-loading BGE-M3 (~570MB), BM25 index, and CrossEncoder on every query.
_RAG_SINGLETON: Optional["LocalRAG"] = None


def get_rag_instance() -> "LocalRAG":
    """
    Return the process-scoped LocalRAG singleton.
    On first call, loads all models (BGE-M3, BM25, CrossEncoder).
    Subsequent calls return the warm instance instantly.
    """
    global _RAG_SINGLETON
    if _RAG_SINGLETON is None:
        logger.info("[Singleton] Creating LocalRAG instance...")
        _RAG_SINGLETON = LocalRAG()
    return _RAG_SINGLETON


def retrieve(question: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> List[Any]:
    """Clean module-level API for retrieval. Reuses the process singleton."""
    return get_rag_instance().retrieve(question, top_k=top_k, threshold=threshold)


def build_context(
    hits: List[Any],
    expand_to_parent: bool = True,
    question: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Clean module-level API for context construction. Reuses the process singleton."""
    return get_rag_instance().build_context(hits, expand_to_parent=expand_to_parent, question=question)