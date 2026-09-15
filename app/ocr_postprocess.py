"""
Engineering OCR Post-Processor for Industrial Documents — SIH 26117.

Corrects OCR errors specific to industrial instrumentation and refinery documents:

1. Equipment tag correction (ISA-5.1): PT-101, FT-204, XV-301
2. Engineering number correction: digit substitution in numeric context
   "1O.25 bar" → "10.25 bar"  (capital O confused with zero)
   "lO kPa"   → "10 kPa"    (lowercase l confused with 1)
3. Unit normalization: canonical SI and process-engineering units
4. Date correction: "12/O8/2026" → "12/08/2026"
5. Confidence-weighted UNVERIFIED tagging for safety-critical values
6. Section heading detection for structured RAG output

All corrections are logged with original → corrected for traceability.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Known engineering units (canonical forms) ─────────────────────────────────
# Maps OCR-error variants → correct form

UNIT_NORMALIZATIONS: Dict[str, str] = {
    # Pressure
    r"\bbar\s*\(\s*g\s*\)":      "bar(g)",
    r"\bBar\b":                   "bar",
    r"\bBAR\b":                   "bar",
    r"\bkpa\b":                   "kPa",
    r"\bKPA\b":                   "kPa",
    r"\bMpa\b":                   "MPa",
    r"\bMPA\b":                   "MPa",
    r"\bpsia\b":                  "psia",
    r"\bPSIA\b":                  "psia",
    r"\bpsig\b":                  "psig",
    r"\bPSIG\b":                  "psig",
    # Temperature
    r"°\s*c\b":                   "°C",
    r"°\s*C\b":                   "°C",
    r"deg\s*c\b":                 "°C",
    r"Deg\s*C\b":                 "°C",
    r"degree\s*c\b":              "°C",
    # Flow
    r"\bm3/h\b":                  "m³/h",
    r"\bM3/H\b":                  "m³/h",
    r"\bNm3/h\b":                 "Nm³/h",
    r"\bnm3/h\b":                 "Nm³/h",
    r"\bl/min\b":                 "L/min",
    r"\bL/MIN\b":                 "L/min",
    # Electrical
    r"\bvdc\b":                   "VDC",
    r"\bVDC\b":                   "VDC",
    r"\bvac\b":                   "VAC",
    r"\bVAC\b":                   "VAC",
    r"\b4\s*-\s*20\s*ma\b":      "4-20 mA",
    r"\b4\s*-\s*20\s*MA\b":      "4-20 mA",
    # Speed
    r"\brpm\b":                   "RPM",
    r"\bRPM\b":                   "RPM",
    r"\bmm/s\b":                  "mm/s",
    r"\bMM/S\b":                  "mm/s",
    # Chemical
    r"\bppm\s*\(v\)":             "ppm(v)",
    r"\bppm\s*\(w\)":             "ppm(w)",
    r"\bPPM\b":                   "ppm",
    r"\bwt\s*%":                  "wt%",
    r"\bWT\s*%":                  "wt%",
}

# ── Regex: numeric measurement context ────────────────────────────────────────
# Matches: number (possibly with O/l substitution errors) followed by a unit
# e.g., "1O.25 bar", "lO kPa", "85.6 °c", "3.42mm/s"

_NUMERIC_UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z])"                              # not preceded by a letter
    r"(\d*[OoIlL\d]+(?:[.,]\d*[OoIlL\d]+)?)"    # number with possible OCR errors
    r"\s*"
    r"(bar(?:\(g\))?|kPa|MPa|°C|°c|psia|psig|"
    r"VDC|VAC|mA|RPM|mm/s|m³/h|Nm³/h|L/min|"
    r"ppm|wt%|mm|cm|m\b|kg|t\b|kW|MW|A\b|V\b)",
    re.IGNORECASE,
)

# ── ISA-5.1 tag prefixes ──────────────────────────────────────────────────────
_TAG_PREFIXES = (
    r"PT|TT|FT|LT|XV|FV|PV|LV|DP|TI|PI|FI|LI|"
    r"ESD|SDV|BDV|MOV|PSV|PRV|CV|FC|FO|AT|AE|"
    r"SE|ST|PDE|FDE|LDE|C|V|E|P|K|R|J|T|H|D"
)

_TAG_PATTERN = re.compile(
    rf"\b({_TAG_PREFIXES})([-\s]?)([0-9OoIlL]{{2,5}})([A-Za-z]?)\b"
)

# ── Date pattern ──────────────────────────────────────────────────────────────
_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[OoIlL]?\d{0,1})"   # day — may have O/l errors
    r"([/\-.])"                        # separator
    r"([OoIlL\d]{1,2})"              # month — may have O/l errors
    r"([/\-.])"                        # separator
    r"(\d{2,4})\b"                    # year
)

# ── Section heading patterns ──────────────────────────────────────────────────
_HEADING_PATTERNS = [
    re.compile(r"^\s*(\d+[\.\d]*)\s+([A-Z][A-Za-z\s\-&/]{3,60})\s*$"),  # "1.2 TRIP SETTINGS"
    re.compile(r"^\s*([A-Z][A-Z\s\-&/]{3,50}:)\s*$"),                    # "ALARM SETTINGS:"
    re.compile(r"^\s*#{1,3}\s+(.+)$"),                                     # Markdown "# Heading"
    re.compile(r"^\s*={3,}\s*(.+?)\s*={3,}\s*$"),                        # "=== Section ==="
]


# ── Correction functions ───────────────────────────────────────────────────────

def _fix_digit_in_measurement(match: re.Match) -> str:
    """
    Fix O→0 and l/I/L→1 substitutions inside a numeric measurement.
    Only applied when the token is followed by a recognized engineering unit.
    """
    number_str, unit_str = match.group(1), match.group(2)
    fixed = ""
    for ch in number_str:
        if ch in ("O", "o"):
            fixed += "0"
        elif ch in ("l", "I", "L") and not fixed.endswith("."):
            # "l" could be a leading digit or mid-number; context: only fix if surrounded by digits
            fixed += "1"
        else:
            fixed += ch
    return f"{fixed} {unit_str}"


def _fix_tag(match: re.Match) -> str:
    """Fix OCR errors inside ISA-5.1 equipment tag numbers."""
    prefix   = match.group(1).upper()
    sep      = "-"  # normalise separator to hyphen
    num_body = match.group(3)
    trailer  = (match.group(4) or "").upper()

    fixed_num = ""
    for ch in num_body:
        if ch in ("O", "o"):
            fixed_num += "0"
        elif ch in ("l", "I"):
            fixed_num += "1"
        else:
            fixed_num += ch

    return f"{prefix}{sep}{fixed_num}{trailer}"


def _fix_date(match: re.Match) -> str:
    """Fix O→0 and l→1 inside date components."""
    def _clean(s: str) -> str:
        return s.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1").replace("L", "1")

    day  = _clean(match.group(1))
    sep1 = match.group(2)
    mon  = _clean(match.group(3))
    sep2 = match.group(4)
    yr   = match.group(5)
    return f"{day}{sep1}{mon}{sep2}{yr}"


def correct_equipment_tags(text: str) -> Tuple[str, List[str]]:
    """
    Apply ISA-5.1 tag correction. Returns (corrected_text, list_of_changes).
    """
    changes = []
    def _replace(m: re.Match) -> str:
        original = m.group(0)
        fixed    = _fix_tag(m)
        if fixed != original:
            changes.append(f"tag: '{original}' → '{fixed}'")
        return fixed

    corrected = _TAG_PATTERN.sub(_replace, text)
    return corrected, changes


def correct_engineering_numbers(text: str) -> Tuple[str, List[str]]:
    """
    Fix digit-substitution errors inside numeric measurements.
    Returns (corrected_text, list_of_changes).
    """
    changes = []
    def _replace(m: re.Match) -> str:
        original = m.group(0)
        fixed    = _fix_digit_in_measurement(m)
        if fixed.replace(" ", "") != original.replace(" ", ""):
            changes.append(f"number: '{original}' → '{fixed}'")
        return fixed

    corrected = _NUMERIC_UNIT_PATTERN.sub(_replace, text)
    return corrected, changes


def normalize_units(text: str) -> Tuple[str, List[str]]:
    """
    Normalize engineering units to their canonical form.
    Returns (corrected_text, list_of_changes).
    """
    changes = []
    result  = text
    for pattern_str, canonical in UNIT_NORMALIZATIONS.items():
        new = re.sub(pattern_str, canonical, result, flags=re.IGNORECASE)
        if new != result:
            changes.append(f"unit → '{canonical}'")
            result = new
    return result, changes


def correct_dates(text: str) -> Tuple[str, List[str]]:
    """
    Fix O/l substitution errors inside dates.
    Returns (corrected_text, list_of_changes).
    """
    changes = []
    def _replace(m: re.Match) -> str:
        original = m.group(0)
        fixed    = _fix_date(m)
        if fixed != original:
            changes.append(f"date: '{original}' → '{fixed}'")
        return fixed

    corrected = _DATE_PATTERN.sub(_replace, text)
    return corrected, changes


def tag_low_confidence_values(
    text: str,
    confidence: float,
    threshold: float = 0.70,
) -> str:
    """
    If overall page OCR confidence is below threshold, append [UNVERIFIED]
    markers to numeric measurements so engineers know to double-check.

    Only tags values that contain a recognized engineering unit.
    """
    if confidence >= threshold:
        return text

    def _mark(m: re.Match) -> str:
        return f"{m.group(0)} [UNVERIFIED]"

    return _NUMERIC_UNIT_PATTERN.sub(_mark, text)


def detect_section_headings(text: str) -> List[Dict[str, str]]:
    """
    Detect section headings in OCR-extracted text using regex heuristics.
    Returns a list of {heading, content} dicts for structured RAG indexing.

    Example output:
    [
        {"heading": "1.2 Trip Settings", "content": "PT-101 HH: 45 bar(g) ..."},
        {"heading": "Alarm Configuration", "content": "..."},
    ]
    """
    lines  = text.splitlines()
    sections: List[Dict[str, str]] = []
    current_heading = ""
    current_lines: List[str] = []

    for line in lines:
        is_heading = False
        for pattern in _HEADING_PATTERNS:
            if pattern.match(line):
                # Save previous section
                if current_heading and current_lines:
                    sections.append({
                        "heading": current_heading.strip(),
                        "content": "\n".join(current_lines).strip(),
                    })
                current_heading = line.strip()
                current_lines   = []
                is_heading = True
                break

        if not is_heading:
            current_lines.append(line)

    # Flush last section
    if current_heading and current_lines:
        sections.append({
            "heading": current_heading.strip(),
            "content": "\n".join(current_lines).strip(),
        })

    return sections


# ── Master post-processor ─────────────────────────────────────────────────────

def postprocess_ocr_text(
    text: str,
    confidence: float = 1.0,
    log_corrections: bool = True,
) -> Dict:
    """
    Run the full engineering OCR post-processing pipeline on extracted text.

    Steps:
      1. Equipment tag correction (ISA-5.1)
      2. Engineering number digit correction
      3. Unit normalization
      4. Date correction
      5. Low-confidence UNVERIFIED tagging
      6. Section heading detection

    Returns:
    {
        "text":         str,              # corrected text
        "sections":     List[dict],       # detected headings + content
        "corrections":  List[str],        # all changes made (for audit log)
        "has_corrections": bool,
    }
    """
    if not text:
        return {"text": "", "sections": [], "corrections": [], "has_corrections": False}

    all_corrections: List[str] = []

    # Step 1: Tag correction
    text, c1 = correct_equipment_tags(text)
    all_corrections.extend(c1)

    # Step 2: Numeric digit correction
    text, c2 = correct_engineering_numbers(text)
    all_corrections.extend(c2)

    # Step 3: Unit normalization
    text, c3 = normalize_units(text)
    all_corrections.extend(c3)

    # Step 4: Date correction
    text, c4 = correct_dates(text)
    all_corrections.extend(c4)

    # Step 5: UNVERIFIED tagging for low-confidence pages
    text = tag_low_confidence_values(text, confidence)

    # Step 6: Section heading detection
    sections = detect_section_headings(text)

    if log_corrections and all_corrections:
        logger.info(
            f"[OCR PostProcess] {len(all_corrections)} correction(s): "
            + " | ".join(all_corrections[:8])
            + (" ..." if len(all_corrections) > 8 else "")
        )

    return {
        "text":            text,
        "sections":        sections,
        "corrections":     all_corrections,
        "has_corrections": bool(all_corrections),
    }
