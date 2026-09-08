"""
Semantic Chunking and Metadata Preservation for SIH 26117.

Features:
- Paragraph- and sentence-aware semantic text splitting (no arbitrary character cutting)
- Stable, unique, deterministic chunk IDs across entire document
- Full preservation of provenance metadata (page_number, location, section, extraction_method, ocr_used)
- Extraction of chunk-level industrial tags (e.g. PT-101, FT-204, XV-301)
- Configurable chunk size, overlap, and minimum chunk size
"""

from pathlib import Path
import re
from typing import Any, Dict, List

from .config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from .document_metadata import classify_document, extract_tags


def split_text_semantically(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> List[str]:
    """
    Split text into semantically cohesive chunks based on paragraph and sentence boundaries.
    Prevents slicing mid-word or mid-sentence.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # 1. Structural Split: Split by Markdown Headers (e.g., # Header, ## Header)
    # We want to keep the header with its content. We split using a lookahead for \n#
    sections = re.split(r"(?=\n#+ )", text)
    
    units: List[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if len(section) <= chunk_size:
            units.append(section)
            continue
            
        # 2. If a section is too large, split by paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        
        for para in paragraphs:
            if len(para) <= chunk_size:
                units.append(para)
            else:
                # 3. If a paragraph is too large, split by sentences
                sentences = re.split(r"(?<=[.?!;])\s+", para)
                current_unit = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(current_unit) + len(s) + 1 <= chunk_size:
                        current_unit = f"{current_unit} {s}".strip() if current_unit else s
                    else:
                        if current_unit:
                            units.append(current_unit)
                        # 4. Fallback: Split long sentences
                        if len(s) > chunk_size:
                            sub_parts = re.findall(rf".{{1,{chunk_size}}}(?:\s+|$)", s)
                            units.extend([sp.strip() for sp in sub_parts if sp.strip()])
                            current_unit = ""
                        else:
                            current_unit = s
                if current_unit:
                    units.append(current_unit)

    # Now assemble units into chunks with overlap
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for unit in units:
        unit_len = len(unit)

        if current_length + unit_len + (1 if current_chunk else 0) <= chunk_size:
            current_chunk.append(unit)
            current_length += unit_len + 1
        else:
            if current_chunk:
                assembled = "\n\n".join(current_chunk).strip()
                if len(assembled) >= min_chunk_size:
                    chunks.append(assembled)

                # Determine overlap context
                overlap_units: List[str] = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) <= chunk_overlap:
                        overlap_units.insert(0, prev)
                        overlap_len += len(prev)
                    else:
                        break
                current_chunk = overlap_units
                current_length = sum(len(u) for u in current_chunk) + len(current_chunk)

            current_chunk.append(unit)
            current_length += unit_len + 1

    if current_chunk:
        assembled = "\n\n".join(current_chunk).strip()
        if assembled:
            if not chunks or assembled != chunks[-1]:
                chunks.append(assembled)

    return chunks


def chunk_documents(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk parsed document records with full metadata and provenance preservation.

    Assigns globally unique and deterministic chunk IDs across the document records.
    """
    chunks: List[Dict[str, Any]] = []
    global_chunk_idx = 0

    for record_idx, record in enumerate(records):
        raw_text = record.get("text", "").strip()
        if not raw_text:
            continue

        source = record.get("source", "Unknown")
        filename = record.get("filename", Path(source).name)
        location = record.get("location", "Unknown")
        page_number = record.get("page_number", 1)
        file_type = record.get("file_type", "unknown")
        section = record.get("section", None)
        is_table = record.get("is_table", False)

        # Baseline classification from path
        doc_meta = classify_document(Path(source))

        # Preserve record-specific overrides (especially OCR and scanned flags)
        content_type = record.get("content_type") or doc_meta.get("content_type", "document")
        category = record.get("category") or doc_meta.get("category", "general")
        document_type = record.get("document_type") or doc_meta.get("document_type", "general")
        extraction_method = record.get("extraction_method", "native_text")
        ocr_used = record.get("ocr_used", False)
        ocr_confidence = record.get("ocr_confidence", None)

        # Parent-Child Hierarchical Context
        parent_id_str = f"p{page_number}_r{record_idx}"
        parent_text = raw_text if len(raw_text) <= 3000 else raw_text[:3000] + "..."

        # For self-contained table rows, preserve them as single chunks if reasonably sized
        if is_table and len(raw_text) <= CHUNK_SIZE * 2:
            text_chunks = [raw_text]
        else:
            # Semantically split record text into precise child chunks
            text_chunks = split_text_semantically(
                raw_text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                min_chunk_size=MIN_CHUNK_SIZE,
            )

        for chunk_offset, chunk_text in enumerate(text_chunks):
            # Extract technical tags specifically from this chunk
            chunk_tags = extract_tags(chunk_text)

            # P0 FIX: Use global_chunk_idx to guarantee absolute uniqueness across all pages/records
            chunk_id_str = f"c{global_chunk_idx}"

            chunk = {
                "text": chunk_text,
                "chunk_id": chunk_id_str,
                "chunk_index": global_chunk_idx,
                "source": source,
                "filename": filename,
                "location": location,
                "page_number": page_number,
                "section": section,
                "file_type": file_type,
                "content_type": content_type,
                "category": category,
                "document_type": document_type,
                "extraction_method": extraction_method,
                "ocr_used": ocr_used,
                "ocr_confidence": ocr_confidence,
                "tags": chunk_tags,
                "is_table": is_table,
                "parent_id": parent_id_str,
                "parent_text": parent_text,
            }

            chunks.append(chunk)
            global_chunk_idx += 1

    return chunks