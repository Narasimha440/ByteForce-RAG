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

    # Basic content type initial guess
    if file_type in {"png", "jpg", "jpeg", "webp"}:
        content_type = "image"
    elif "scan" in filename or "scanned" in filename:
        content_type = "scanned_document"
    else:
        content_type = "document"

    # Detect industrial document types
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

    # Broad category mapping
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
    revision = extract_revision(filename=path.name)

    return {
        "file_type": file_type,
        "content_type": content_type,
        "category": category,
        "document_type": document_type,
        "revision": revision,
    }


def extract_revision(text: str = "", filename: str = "") -> str:
    """
    Extract engineering document revision (e.g., 'Rev 0', 'Rev 1', 'Rev A', 'Rev 2.1', 'Draft', 'Superseded').
    Returns 'Active' if no specific revision code is found.
    """
    combined = f"{filename} {text[:1500]}"
    if re.search(r"\b(superseded|obsolete|withdrawn)\b", combined, re.IGNORECASE):
        return "Superseded"
    if re.search(r"\b(draft|preliminary|unapproved)\b", combined, re.IGNORECASE):
        return "Draft"

    match = re.search(r"\b(?:Rev|Revision)[\s.:_-]*([0-9]+(?:\.[0-9]+)?|[A-Z])\b", combined, re.IGNORECASE)
    if match:
        return f"Rev {match.group(1).upper()}"
    return "Active"



def extract_tags(text: str) -> List[str]:
    """
    Extract accurate industrial and technical tags from document text.

    Preserves exact punctuation and format (e.g. PT-101, FT-204, XV-301, R0S03).
    Does NOT aggressively normalize, strip hyphens, or invent tags.
    """
    if not text:
        return []

    tags: Set[str] = set()

    # Pattern 1: Hyphenated Instrument & Equipment Tags (e.g., PT-101, FT-204, XV-301, ICP-01, P-101A)
    # 1 to 5 uppercase letters, hyphen, 1 to 5 digits, optional single letter suffix
    hyphenated = re.findall(r"\b[A-Z]{1,5}-\d{1,5}[A-Z]?\b", text)
    tags.update(hyphenated)

    # Pattern 2: Industrial Rack/Slot tags (e.g. R0S03, R1S05)
    rack_slot = re.findall(r"\bR\d+S\d+\b", text)
    tags.update(rack_slot)

    # Pattern 3: Unhyphenated Equipment Codes with at least 2 letters and 2 digits (e.g. PT101, PLC01)
    unhyphenated = re.findall(r"\b[A-Z]{2,4}\d{2,4}[A-Z]?\b", text)
    # Exclude common non-tag acronyms like ISO9001
    tags.update(unhyphenated)

    # Pattern 4: Industrial Signal & Voltage tags (e.g. 24VDC, 120VAC, 4-20mA)
    signals = re.findall(r"\b(?:24\s*VDC|120\s*VAC|230\s*VAC|480\s*VAC|4-20\s*mA)\b", text, re.IGNORECASE)
    tags.update(s.upper() for s in signals)

    # Pattern 5: Critical industrial acronyms when present in uppercase
    acronyms = re.findall(r"\b(?:P&ID|SCADA|MAH|SOP|FAT|ICP|BOM|PLC|RTU)\b", text)
    tags.update(acronyms)

    # Filter out common false positives
    ignore = {"PAGE", "SHEET", "DOC", "PDF", "ITEM", "REV", "NOTE", "DATE"}
    filtered_tags = [t for t in tags if t.upper() not in ignore]

    return sorted(filtered_tags)


# Backwards compatibility alias
extract_industrial_tags = extract_tags

def extract_measurements(text: str) -> List[str]:
    """
    Extract engineering measurements (value + unit) from text.
    Matches numbers followed by common industrial units.
    """
    if not text:
        return []
        
    measurements = set()
    
    # Common industrial units (case-insensitive for some, specific for others)
    # Using a broad pattern to catch variations
    units_pattern = r"(?:bar|barg|psig|kpa|mpa|psi|°C|°F|deg C|deg F|mm/s|m/s|rpm|hz|v|kv|ma|a|kw|mw|m3/h|tph|kg/h|lpm)"
    
    # Matches integers or decimals followed by optional space and then the unit
    pattern = rf"\b(\d+(?:\.\d+)?)\s*({units_pattern})\b"
    
    matches = re.finditer(pattern, text, re.IGNORECASE)
    for match in matches:
        value = match.group(1)
        # Normalize unit spacing
        unit = match.group(2).lower()
        if unit == "deg c": unit = "°C"
        elif unit == "deg f": unit = "°F"
        elif unit == "°c": unit = "°C"
        elif unit == "°f": unit = "°F"
        elif unit == "kpa": unit = "kPa"
        elif unit == "mpa": unit = "MPa"
        elif unit == "mw": unit = "MW"
        elif unit == "kw": unit = "kW"
        elif unit == "kv": unit = "kV"
        elif unit == "hz": unit = "Hz"
        
        measurements.add(f"{value} {unit}")
        
    return sorted(list(measurements))