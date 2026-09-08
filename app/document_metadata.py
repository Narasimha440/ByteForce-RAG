"""
Industrial Metadata Classification and Technical Tag Extraction for SIH 26117.

Categorizes industrial documents into engineering, compliance, standards,
templates, and general; identifies document types (pid, inspection, instrumentation,
control_panel, scada, etc.); extracts technical instrument/equipment tags without
destructive normalization.
"""

from pathlib import Path
import re
from typing import Dict, List, Set


def classify_document(path: Path) -> Dict[str, str]:
    """
    Generate core metadata for a document based on filename and file path.
    Does not hardcode rigid assumptions; allows parser/OCR to refine content_type.
    """
    filename = path.name.lower()
    path_text = str(path).lower()
    file_type = path.suffix.lower().lstrip(".")

    
    if file_type in {"png", "jpg", "jpeg", "webp"}:
        content_type = "image"
    elif "scan" in filename or "scanned" in filename:
        content_type = "scanned_document"
    else:
        content_type = "document"

    
    if any(term in path_text or term in filename for term in ["p&id", "p_and_i", "pid", "legend"]):
        document_type = "pid"

    elif any(term in filename for term in [
        "inspection",
        "checklist",
        "accident",
        "investigation",
        "audit",
    ]):
        document_type = "inspection"

    elif any(term in filename for term in [
        "instrument",
        "instrumentation",
        "mounting",
        "transmitter",
        "sensor",
    ]):
        document_type = "instrumentation"

    elif any(term in filename for term in [
        "control_panel",
        "control-panel",
        "panel",
        "icp",
        "wiring",
        "power distribution",
    ]):
        document_type = "control_panel"

    elif any(term in filename for term in [
        "network",
        "architecture",
        "topology",
    ]):
        document_type = "network"

    elif any(term in filename for term in [
        "scada",
        "hmi",
        "plc",
    ]):
        document_type = "scada"

    elif any(term in filename for term in [
        "standard",
        "manual",
        "specification",
        "sop",
    ]):
        document_type = "standard"

    elif any(term in filename for term in [
        "template",
        "form",
        "inventory",
    ]):
        document_type = "template"

    else:
        document_type = "general"

    
    category_map = {
        "pid": "engineering",
        "instrumentation": "engineering",
        "control_panel": "engineering",
        "network": "engineering",
        "scada": "engineering",
        "inspection": "compliance",
        "standard": "standards",
        "template": "templates",
        "general": "general",
    }

    category = category_map.get(document_type, "general")

    return {
        "file_type": file_type,
        "content_type": content_type,
        "category": category,
        "document_type": document_type,
    }


def extract_tags(text: str) -> List[str]:
    """
    Extract accurate industrial and technical tags from document text.

    Preserves exact punctuation and format (e.g. PT-101, FT-204, XV-301, R0S03).
    Does NOT aggressively normalize, strip hyphens, or invent tags.
    """
    if not text:
        return []

    tags: Set[str] = set()

    
    
    hyphenated = re.findall(r"\b[A-Z]{1,5}-\d{1,5}[A-Z]?\b", text)
    tags.update(hyphenated)

    
    rack_slot = re.findall(r"\bR\d+S\d+\b", text)
    tags.update(rack_slot)

    
    unhyphenated = re.findall(r"\b[A-Z]{2,4}\d{2,4}[A-Z]?\b", text)
    
    tags.update(unhyphenated)

    
    
    signals = re.findall(r"\b(?:24\s*VDC|120\s*VAC|230\s*VAC|480\s*VAC|4-20\s*mA)\b", text, re.IGNORECASE)
    tags.update(s.upper() for s in signals)

    
    acronyms = re.findall(r"\b(?:P&ID|SCADA|MAH|SOP|FAT|ICP|BOM|PLC|RTU)\b", text)
    tags.update(acronyms)

    
    ignore = {"PAGE", "SHEET", "DOC", "PDF", "ITEM", "REV", "NOTE", "DATE"}
    filtered_tags = [t for t in tags if t.upper() not in ignore]

    return sorted(filtered_tags)



extract_industrial_tags = extract_tags