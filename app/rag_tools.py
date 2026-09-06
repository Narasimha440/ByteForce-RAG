"""
Clean Agentic Tool Interfaces for SIH 26117.

Problem Statement Requirement:
"The assistant also needs to actually act like an agent. Plan out multi step
work, call local tools such as file read and write, code execution in a sandbox,
spreadsheet work, internal document search, and iterate on a task instead of
answering once and stopping."

This module provides:
1. Standardized Python functions for the agent orchestrator (Qwen3).
2. JSON schemas in standard OpenAI/Qwen tool calling format for drop-in registration.
3. Strict air-gap verification and source-grounded evidence returns.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .airgap import audit_local_event, verify_airgap_status
from .document_metadata import extract_industrial_tags
from .document_registry import DocumentRegistry
from .ocr import extract_text_with_ocr
from .rag import LocalRAG

logger = logging.getLogger(__name__)

# Reusable RAG singleton for tool calls
_rag_instance: Optional[LocalRAG] = None


def get_rag() -> LocalRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LocalRAG()
    return _rag_instance


def search_internal_knowledge(
    query: str,
    top_k: int = 5,
    document_type: Optional[str] = None,
) -> str:
    """
    Search the sovereign on-premise knowledge base for confidential refinery procedures,
    SOPs, crude assays, equipment manuals, and inspection reports.

    Args:
        query: Natural language query or equipment tag (e.g. 'BDV-701 operating limits', 'HCU quench').
        top_k: Number of relevant evidence chunks to retrieve.
        document_type: Optional filter by category (e.g. '01_Standards_Reference', '09_Inspection_Accident').

    Returns:
        Formatted evidence context with [EVIDENCE N] citations, document name, page numbers, and tags.
    """
    audit_local_event(
        component="RAG_TOOLS",
        operation="search_internal_knowledge",
        details=f"Query: {query[:60]} | top_k={top_k}",
    )

    filter_criteria = None
    if document_type:
        filter_criteria = {"document_type": document_type}

    rag = get_rag()
    hits = rag.retrieve(query, top_k=top_k, filter_criteria=filter_criteria)

    if not hits:
        return "No relevant evidence found in the local sovereign knowledge base for this query."

    context, sources = rag.build_context(hits, question=query)
    return context


def lookup_equipment_tag(tag: str) -> Dict[str, Any]:
    """
    Look up technical specifications, operating limits, calibration records, and P&ID
    references for a specific industrial equipment tag (e.g. 'BDV-701', 'PT-101', 'XV-301').

    Args:
        tag: The alphanumeric instrument or equipment tag name.

    Returns:
        Structured dictionary with asset dossier, tag occurrences, parent documents, and relevant excerpt.
    """
    clean_tag = tag.strip().upper()
    audit_local_event(
        component="RAG_TOOLS",
        operation="lookup_equipment_tag",
        details=f"Tag lookup: {clean_tag}",
    )

    rag = get_rag()
    hits = rag.retrieve(clean_tag, top_k=5)

    from .equipment_graph import get_equipment_dossier
    dossier = get_equipment_dossier(clean_tag, hits)

    matching_chunks = []
    for hit in hits:
        text = hit.get("text", "")
        tags = hit.get("tags", [])
        if clean_tag in [t.upper() for t in tags] or clean_tag in text.upper():
            matching_chunks.append({
                "document": hit.get("filename"),
                "page": hit.get("page_number"),
                "category": hit.get("document_type"),
                "score": hit.get("score"),
                "excerpt": text[:300] + "..." if len(text) > 300 else text,
            })

    return {
        "tag": clean_tag,
        "dossier": dossier,
        "matches_found": len(matching_chunks),
        "references": matching_chunks,
    }


def inspect_scanned_document(file_path: str, page_number: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract text and industrial equipment tags from a scanned inspection report or drawing
    using local on-device RapidOCR (ONNX). Zero data leaves the machine.

    Args:
        file_path: Absolute or relative path to the PDF or image file.
        page_number: Optional 1-based page number to inspect. If None, inspects the document.

    Returns:
        Dictionary containing extracted OCR text, detected industrial equipment tags, and character count.
    """
    target_path = Path(file_path)
    if not target_path.exists():
        return {"error": f"File not found: {file_path}", "success": False}

    audit_local_event(
        component="RAG_TOOLS",
        operation="inspect_scanned_document",
        details=f"Local OCR inspection on: {target_path.name}",
    )

    extracted_text, ocr_engine = extract_text_with_ocr(target_path, page_num=page_number)
    tags = extract_industrial_tags(extracted_text)

    return {
        "success": True,
        "file_name": target_path.name,
        "ocr_engine_used": ocr_engine,
        "character_count": len(extracted_text),
        "detected_industrial_tags": tags,
        "extracted_text": extracted_text,
    }


def list_registered_documents() -> List[Dict[str, Any]]:
    """
    List all documents currently ingested and indexed in the local sovereign knowledge base.
    """
    audit_local_event(
        component="RAG_TOOLS",
        operation="list_registered_documents",
        details="Registry lookup",
    )
    registry = DocumentRegistry()
    docs = registry.list_documents()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "category": doc.document_type,
            "chunk_count": doc.chunk_count,
            "ocr_used": doc.ocr_used,
            "ingested_at": doc.ingested_at,
        }
        for doc in docs
    ]


def get_airgap_status() -> Dict[str, Any]:
    """
    Verify and return the air-gap compliance status of the workbench, proving
    that zero external network calls have been made.
    """
    return verify_airgap_status()


# ==============================================================================
# Standard Tool Schemas (OpenAI / Qwen Function Calling Specification)
# ==============================================================================
AGENT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_internal_knowledge",
            "description": (
                "Search the sovereign on-premise knowledge base for confidential refinery procedures, "
                "SOPs, crude assays, equipment manuals, and inspection reports using hybrid retrieval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or keywords (e.g. 'Hydrocracker quench valve operating pressure')",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of evidence chunks to retrieve (default: 5)",
                    },
                    "document_type": {
                        "type": "string",
                        "description": "Optional category filter (e.g. '01_Standards_Reference', '09_Inspection_Accident')",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_equipment_tag",
            "description": (
                "Look up technical specifications, operating limits, and calibration records for a "
                "specific industrial equipment tag (e.g. 'BDV-701', 'PT-101', 'XV-301')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "The exact alphanumeric instrument or valve tag (e.g. 'BDV-701')",
                    },
                },
                "required": ["tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_scanned_document",
            "description": (
                "Extract text and industrial equipment tags from a scanned inspection report or drawing "
                "using local on-device RapidOCR. Completely sovereign, zero cloud transmission."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the scanned document or image file",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Optional 1-based page number to inspect",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_registered_documents",
            "description": "List all confidential documents, SOPs, spreadsheets, and drawings registered in the local RAG repository.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_airgap_status",
            "description": "Verify air-gap compliance and retrieve proof that zero external outbound network calls have occurred.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]
