"""
Refinery Domain Ontology & Query Expansion Engine for MRPL (SIH 26117).

Bridges the vocabulary gap between plant engineers' shorthand abbreviations
and formal refinery engineering documentation, SOPs, and standards.

Provides:
1. MRPL Process Unit Ontology (CDU, VDU, HCU, FCCU, DCU, HGU, SRU, SPM, DHDT, CCR, PPU, CPP).
2. Industrial Safety & Systems Ontology (EDS, ESD, SIL-3, MAH, OISD, BDV, PSV, PRV, PLEM).
3. Fuel & Refinery Products Ontology (BS-VI, HSD, MS, ATF, RVP, TAN, API).
4. Dense Query Augmentation (for semantic BGE-M3 embedding).
5. Lexical Boost Token Generation (for Okapi BM25 precision).
"""

import re
from typing import Dict, List, Set, Tuple

# Comprehensive MRPL & Petroleum Refinery Domain Knowledge Base
REFINERY_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    # Process Units
    "CDU": {
        "full_name": "Crude Distillation Unit",
        "synonyms": ["CDU-1", "CDU-2", "CDU-3", "atmospheric distillation", "crude tower", "atmospheric residue"],
        "context": "Atmospheric distillation separating raw crude into naphtha, kerosene, gas oil, and atmospheric residue.",
    },
    "VDU": {
        "full_name": "Vacuum Distillation Unit",
        "synonyms": ["VDU-1", "VDU-2", "VDU-3", "vacuum column", "HVGO", "vacuum residue", "VR"],
        "context": "Deep cut vacuum distillation processing atmospheric residue into heavy vacuum gas oil and vacuum residue.",
    },
    "HCU": {
        "full_name": "Hydrocracker Unit",
        "synonyms": ["HCU-1", "HCU-2", "hydrocracking", "reactor loop", "quench gas", "QCV-101"],
        "context": "High-pressure catalytic hydrocracking unit operating at 165-180 bar for converting vacuum gas oils into BS-VI diesel and ATF.",
    },
    "FCCU": {
        "full_name": "Fluidized Catalytic Cracking Unit",
        "synonyms": ["PFCCU", "cat cracker", "regenerator", "spent catalyst", "riser reactor", "XV-301"],
        "context": "Petrochemical fluidized catalytic cracking converting heavy gas oils into high-octane gasoline, propylene, and LPG.",
    },
    "DCU": {
        "full_name": "Delayed Coker Unit",
        "synonyms": ["delayed coking", "coke drums", "petcoke", "coker gas oil"],
        "context": "Thermal cracking unit processing vacuum residue into petroleum coke and lighter distillates.",
    },
    "HGU": {
        "full_name": "Hydrogen Generation Unit",
        "synonyms": ["HGU-1", "steam methane reformer", "SMR", "hydrogen makeup"],
        "context": "Steam reforming unit producing high-purity hydrogen required for hydrocracker and hydrotreater desulfurization.",
    },
    "SRU": {
        "full_name": "Sulphur Recovery Unit",
        "synonyms": ["SRU-1", "Claus plant", "tail gas treating unit", "TGTU", "elemental sulfur"],
        "context": "Sulfur recovery plant converting toxic hydrogen sulfide (H2S) acid gas into pure molten sulfur.",
    },
    "SPM": {
        "full_name": "Single Point Mooring",
        "synonyms": ["offshore terminal", "crude unloading", "submarine pipeline", "PLEM", "PRV-901"],
        "context": "Offshore crude tanker unloading buoy located 16.5 km off Mangalore coast connected via 48-inch subsea pipeline.",
    },
    "DHDT": {
        "full_name": "Diesel Hydrotreater",
        "synonyms": ["DHDS", "diesel desulfurization", "ULSD", "ultra-low sulfur diesel"],
        "context": "Catalytic hydrotreating unit reducing sulfur content in diesel pool to under 10 ppm for BS-VI compliance.",
    },
    "CCR": {
        "full_name": "Continuous Catalytic Regeneration Platformer",
        "synonyms": ["CCR-1", "CCR-2", "catalytic reformer", "high-octane MS", "reformate"],
        "context": "Catalytic reforming unit converting low-octane heavy naphtha into high-octane motor gasoline blendstock.",
    },
    "PPU": {
        "full_name": "Polypropylene Plant",
        "synonyms": ["Mangpol", "propylene polymerization", "polypropylene pellets"],
        "context": "440 KTPA petrochemical plant converting polymer-grade propylene from PFCCU into polypropylene homopolymers and copolymers.",
    },

    # Safety, Interlocks & Critical Systems
    "EDS": {
        "full_name": "Emergency Depressurization System",
        "synonyms": ["HPE", "high pressure depressuring", "BDV-701", "flare blowdown", "rapid depressuring"],
        "context": "Emergency system that rapidly depressurizes the high-pressure reactor loop to flare header during runaway exotherm or hydrogen leak.",
    },
    "ESD": {
        "full_name": "Emergency Shutdown System",
        "synonyms": ["ESD-501", "safety interlock", "emergency trip", "feed cut"],
        "context": "Automated safety instrumented system that trips feed pumps, isolates vessels, and initiates safe plant shutdown.",
    },
    "SIL-3": {
        "full_name": "Safety Integrity Level 3",
        "synonyms": ["SIL3", "SIL", "stroke time limit", "proof test", "3.0 seconds"],
        "context": "Mandatory safety integrity requirement demanding 3.0 second maximum closing stroke time for emergency isolation valves like XV-301.",
    },
    "MAH": {
        "full_name": "Major Accident Hazard",
        "synonyms": ["MAH factory", "statutory safety audit", "hazard identification", "safety report"],
        "context": "Regulatory classification under Factory Rules for high-risk installations storing hazardous hydrocarbons.",
    },
    "OISD": {
        "full_name": "Oil Industry Safety Directorate",
        "synonyms": ["OISD-156", "OISD standard", "fire protection", "refinery safety standard"],
        "context": "Government of India regulatory body formulating safety and loss prevention standards for the oil and gas industry.",
    },
    "BDV": {
        "full_name": "Blowdown Valve",
        "synonyms": ["BDV-701", "emergency depressurization valve", "depressuring valve"],
        "context": "Remotely operated failsafe emergency depressurization valve venting high-pressure gas to the flare.",
    },
    "PSV": {
        "full_name": "Pressure Safety Valve",
        "synonyms": ["safety valve", "relief valve", "PRV", "pressure relief", "set pressure", "overpressure protection"],
        "context": "Spring-loaded or pilot-operated autonomous safety valve protecting vessels and piping against catastrophic overpressure.",
    },
    "MOV": {
        "full_name": "Motor Operated Valve",
        "synonyms": ["motorized valve", "electric actuator", "MOV-104", "isolation valve", "remote operated valve"],
        "context": "Heavy-duty electric actuator driven valve used for pipeline routing, tank farm isolation, and main line control.",
    },
    "XV": {
        "full_name": "Emergency Isolation Valve",
        "synonyms": ["XV-301", "shutdown valve", "emergency shutdown valve", "SDV", "failsafe valve", "SIL-3 valve"],
        "context": "Fail-safe automated fast-acting isolation valve designed to seal off hydrocarbon process units within 3 seconds under SIL-3 trip conditions.",
    },
    "PT": {
        "full_name": "Pressure Transmitter",
        "synonyms": ["pressure sensor", "PT-101", "pressure transducer", "process pressure", "gauge pressure"],
        "context": "Electronic instrument measuring line or vessel static/differential pressure and outputting 4-20 mA / HART signal to DCS.",
    },
    "TT": {
        "full_name": "Temperature Transmitter",
        "synonyms": ["temperature sensor", "RTD", "thermocouple", "TT-101", "process temperature"],
        "context": "Temperature measurement instrument converting RTD/thermocouple millivolts into calibrated temperature readings.",
    },
    "FT": {
        "full_name": "Flow Transmitter",
        "synonyms": ["flow meter", "FT-204", "orifice plate", "Coriolis flow meter", "mass flow rate"],
        "context": "Process flow measurement transmitter monitoring volumetric or mass flow of hydrocarbons, steam, or cooling water.",
    },
    "LT": {
        "full_name": "Level Transmitter",
        "synonyms": ["level sensor", "radar level gauge", "differential pressure level", "LT-101", "liquid level"],
        "context": "Continuous liquid level transmitter measuring tank or distillation column bottom levels.",
    },
    "HIPPS": {
        "full_name": "High Integrity Pressure Protection System",
        "synonyms": ["overpressure protection", "SIL-3", "SIL-4", "fast shutdown", "subsea HIPPS"],
        "context": "Safety instrumented system acting as the last line of defense against overpressure before mechanical relief valves activate.",
    },
    "SCADA": {
        "full_name": "Supervisory Control and Data Acquisition",
        "synonyms": ["telemetry", "human machine interface", "HMI", "remote terminal unit", "RTU"],
        "context": "Plantwide telemetry and supervisory control architecture linking remote terminal units with central engineering workstations.",
    },
    "DCS": {
        "full_name": "Distributed Control System",
        "synonyms": ["process automation", "Yokogawa DCS", "Honeywell DCS", "operator workstation"],
        "context": "Multi-loop digital process control system coordinating refinery unit operations, cascade loops, and alarms.",
    },
    "SIS": {
        "full_name": "Safety Instrumented System",
        "synonyms": ["safety interlock", "emergency shutdown logic", "Triconex", "SIL logic solver"],
        "context": "Dedicated failsafe logic solver executing emergency trip routines independent of basic process control.",
    },
    "PLEM": {
        "full_name": "Pipeline End Manifold",
        "synonyms": ["subsea manifold", "SPM pipeline manifold", "subsea isolation valve"],
        "context": "Subsea manifold structure anchoring the offshore SPM marine hose to the 48-inch pipeline to MRPL shore tank farm.",
    },

    # Products & Quality Specifications
    "BS-VI": {
        "full_name": "Bharat Stage VI Fuel Standard",
        "synonyms": ["BS6", "BS-6", "10 ppm sulfur", "clean fuel", "ultra-low sulfur"],
        "context": "Mandatory national clean fuel standard capping sulfur content at 10 mg/kg (ppm) max for high speed diesel and motor gasoline.",
    },
    "HSD": {
        "full_name": "High Speed Diesel",
        "synonyms": ["gas oil", "automotive diesel", "cetane index 48", "diesel pool"],
        "context": "Refined diesel fuel meeting BS-VI specifications: minimum 48 cetane index, flash point > 38°C, sulfur < 10 ppm.",
    },
    "MS": {
        "full_name": "Motor Spirit",
        "synonyms": ["petrol", "gasoline", "RON 91", "RON 95", "octane"],
        "context": "High-octane automotive gasoline produced from CCR reformate and PFCCU cracked gasoline.",
    },
    "ATF": {
        "full_name": "Aviation Turbine Fuel",
        "synonyms": ["jet fuel", "aviation kerosene", "Jet A-1", "smoke point 21 mm"],
        "context": "High-purity kerosene cut for commercial aircraft: smoke point > 21 mm, freezing point < -47°C, flash point > 38°C.",
    },
    "RVP": {
        "full_name": "Reid Vapor Pressure",
        "synonyms": ["vapor pressure", "volatility", "naphtha RVP", "0.85 bar limit"],
        "context": "Measure of fuel volatility, critical for preventing vapor lock in engines and storage tank losses.",
    },
    "TAN": {
        "full_name": "Total Acid Number",
        "synonyms": ["acid number", "naphthenic acid", "mg KOH/g", "crude corrosivity"],
        "context": "Metric of naphthenic acid content in crude oil; values above 0.5 mg KOH/g require specialized metallurgy.",
    },
    "API": {
        "full_name": "API Gravity",
        "synonyms": ["crude density", "specific gravity", "light crude", "heavy crude"],
        "context": "Standard petroleum scale measuring crude oil density; higher values indicate lighter, more valuable crudes.",
    },
}

# Equipment Tag Regex
EQUIPMENT_TAG_REGEX = re.compile(
    r"\b([A-Z]{1,5}-\d{1,5}[A-Z]?|[A-Z]{2,4}\d{2,4}[A-Z]?)\b"
)


def extract_query_equipment_tags(query: str) -> List[str]:
    """
    Extract specific equipment tags mentioned in the query (e.g., BDV-701, XV-301, PT-101, V-102).
    """
    if not query:
        return []
    matches = EQUIPMENT_TAG_REGEX.findall(query.upper())
    # Exclude ontology acronyms like CDU, VDU, HCU from being considered single equipment tags
    filtered = [m for m in matches if m not in REFINERY_ONTOLOGY and len(m) >= 4]
    return sorted(list(set(filtered)))


def expand_refinery_query(query: str) -> Tuple[str, List[str]]:
    """
    Expands an industrial user query using MRPL domain ontology.

    Returns:
        Tuple of (dense_augmented_query, lexical_boost_tokens)
        - dense_augmented_query: query string enriched with full unit names & engineering definitions.
        - lexical_boost_tokens: list of high-value synonyms and tags for BM25 boosting.
    """
    if not query or not query.strip():
        return query, []

    clean_query = query.strip()
    words = re.findall(r"\b[A-Za-z0-9_\-\./]+\b", clean_query)
    upper_words = [w.upper() for w in words]

    matched_definitions = []
    lexical_boost_tokens: Set[str] = set()

    for word in upper_words:
        # Check direct ontology match
        if word in REFINERY_ONTOLOGY:
            entry = REFINERY_ONTOLOGY[word]
            matched_definitions.append(f"{word} ({entry['full_name']}): {entry['context']}")
            lexical_boost_tokens.add(entry["full_name"])
            for syn in entry["synonyms"]:
                lexical_boost_tokens.add(syn)

        # Check hyphenated variations (e.g. BS6 -> BS-VI)
        elif word == "BS6" or word == "BSVI":
            entry = REFINERY_ONTOLOGY["BS-VI"]
            matched_definitions.append(f"BS-VI ({entry['full_name']}): {entry['context']}")
            lexical_boost_tokens.add("BS-VI")
            lexical_boost_tokens.add("Bharat Stage VI")

    # Also detect equipment tags in query and boost them
    tags = extract_query_equipment_tags(clean_query)
    for tag in tags:
        lexical_boost_tokens.add(tag)

    if not matched_definitions:
        return clean_query, sorted(list(lexical_boost_tokens))

    # Dense query augmentation: original query + domain context
    augmented_query = f"{clean_query} | Context: " + " ".join(matched_definitions[:2])
    return augmented_query, sorted(list(lexical_boost_tokens))


# Mapping from natural language engineering phrases to standard plant codes
SPOKEN_PHRASE_TO_CODE: Dict[str, str] = {
    "blowdown valve": "BDV",
    "blow down valve": "BDV",
    "depressurizing valve": "BDV",
    "pressure safety valve": "PSV",
    "relief valve": "PSV",
    "safety relief valve": "PSV",
    "emergency isolation valve": "XV",
    "emergency shutdown valve": "XV",
    "isolation valve": "XV",
    "motor operated valve": "MOV",
    "motorized valve": "MOV",
    "pressure transmitter": "PT",
    "pressure sensor": "PT",
    "pressure gauge": "PT",
    "temperature transmitter": "TT",
    "temperature sensor": "TT",
    "flow transmitter": "FT",
    "flow meter": "FT",
    "level transmitter": "LT",
    "level gauge": "LT",
    "emergency shutdown": "ESD",
    "emergency depressurization": "EDS",
    "rapid depressuring": "EDS",
}


def expand_equipment_acronyms(query: str) -> str:
    """
    Bidirectional acronym expansion:
    1. If user uses full spoken phrase ("blowdown valve"), appends exact code ("BDV").
    2. If user uses exact code ("BDV"), appends full engineering name ("Blowdown Valve").
    Returns the enriched query string.
    """
    if not query:
        return ""
    q_lower = query.lower()
    appends = []

    # 1. Natural phrase to code
    for phrase, code in SPOKEN_PHRASE_TO_CODE.items():
        if phrase in q_lower and code not in query.upper():
            appends.append(code)

    # 2. Acronym to full name
    q_words = re.findall(r"\b[A-Za-z0-9_-]+\b", query.upper())
    for word in q_words:
        if word in REFINERY_ONTOLOGY:
            full = REFINERY_ONTOLOGY[word]["full_name"]
            if full.lower() not in q_lower:
                appends.append(full)

    if appends:
        return f"{query.strip()} ({' '.join(set(appends))})"
    return query.strip()

