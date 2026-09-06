"""
Equipment Knowledge Graph & Relational Tag Cross-Referencing for MRPL (SIH 26117).

Connects industrial equipment tags (BDV-701, XV-301, PT-101, FT-204, V-102, PRV-901)
across disparate files: P&IDs, Operating SOPs, Safety Audits, and Instrument Indices.

Provides:
1. Seeded MRPL Equipment Technical Dossiers with verified trip points, units, and failure modes.
2. Dynamic Tag Miner that constructs on-the-fly dossiers from indexed document evidence.
3. Formatted Industrial Equipment Dossier Box for high-confidence LLM context injection.
"""

from typing import Any, Dict, List, Optional

# Verified MRPL Equipment Master Dossiers
MRPL_EQUIPMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "BDV-701": {
        "tag": "BDV-701",
        "unit": "Hydrocracker Unit (HCU-1 / HCU-2)",
        "equipment_type": "Emergency High-Pressure Depressurization Blowdown Valve",
        "service": "Reactor Loop High Pressure Gas to Flare Header",
        "normal_operating_range": "165 – 180 bar(g) (Valve Normally Closed)",
        "trip_limit": "> 188 bar(g) or Catalyst Bed Delta-T > 45°C",
        "failsafe_action": "Fail Open (FO) to Flare Knockout Drum V-102 on Emergency Depressurization System (EDS) activation",
        "relief_destination": "Refinery High Pressure Flare Header (15-minute depressurization window to < 7 bar)",
        "referenced_documents": [
            "MRPL_HCU_Hydrocracker_Operating_SOP.docx",
            "MRPL_Instrumentation_Master_Index.xlsx",
        ],
    },
    "XV-301": {
        "tag": "XV-301",
        "unit": "Petrochemical Fluidized Catalytic Cracking Unit (PFCCU)",
        "equipment_type": "Spent Catalyst Slide Valve / Emergency Isolation Valve",
        "service": "Regenerator Dense Bed to Riser Reactor Catalyst Circulation",
        "normal_operating_range": "35% – 65% Open (Regenerator Bed: 680°C – 715°C)",
        "trip_limit": "Mandatory SIL-3 Maximum Closing Stroke Time: 3.0 seconds",
        "failsafe_action": "Fail Close (FC) on low combustion air pressure (< 2.2 bar) or ESD-501 trip",
        "statutory_audit_finding": (
            "Annual proof test recorded stroke time of 5.8 seconds (EXCEEDS SIL-3 3.0s limit). "
            "Particulate residue in 24 VDC solenoid dump port. Corrective action: flush hydraulic fluid, "
            "replace dual solenoid pilot assembly, re-test within 48 hours (MRPL Safety Audit 2026)."
        ),
        "referenced_documents": [
            "MRPL_FCCU_Operating_Manual.docx",
            "MRPL_MAH_Factory_Annual_Safety_Audit_2026.pdf",
            "MRPL_Instrumentation_Master_Index.xlsx",
        ],
    },
    "PT-101": {
        "tag": "PT-101",
        "unit": "PFCCU / Combustion Air Blower",
        "equipment_type": "Combustion Air Discharge Pressure Transmitter",
        "service": "Main Air Blower Discharge Header to Regenerator Grid",
        "normal_operating_range": "2.8 – 3.4 bar(g) (4-20 mA HART signal)",
        "trip_limit": "Low Trip < 2.2 bar(g) (Auto-trips catalyst circulation to prevent hydrocarbon reversal)",
        "failsafe_action": "Initiates interlock to close spent catalyst slide valve XV-301",
        "referenced_documents": [
            "MRPL_FCCU_Operating_Manual.docx",
            "MRPL_Instrumentation_Master_Index.xlsx",
        ],
    },
    "FT-204": {
        "tag": "FT-204",
        "unit": "Crude Distillation Unit (CDU) / Furnace F-101",
        "equipment_type": "Differential Pressure Pass Flow Transmitter",
        "service": "F-101 Crude Charge Pass 4 Balanced Flow Monitoring",
        "normal_operating_range": "180 – 240 m3/h per pass",
        "trip_limit": "Low-Low Flow < 120 m3/h (Risk of furnace tube coking and rupture)",
        "statutory_audit_finding": (
            "Field inspection detected moisture ingress into terminal block due to degraded cable gland. "
            "Cleaned, desiccant added, and certified IP66 cable gland installed (MRPL Safety Audit 2026)."
        ),
        "referenced_documents": [
            "MRPL_MAH_Factory_Annual_Safety_Audit_2026.pdf",
            "MRPL_Instrumentation_Master_Index.xlsx",
        ],
    },
    "V-102": {
        "tag": "V-102",
        "unit": "Refinery Offsites & Flare Relief System",
        "equipment_type": "Emergency Flare Knockout Drum",
        "service": "Liquid Droplet Separation from Relief Gases prior to Flare Stack",
        "normal_operating_range": "0.1 – 0.5 bar(g), Ambient Temperature",
        "trip_limit": "High-High Liquid Level > 75% (Risk of burning liquid carryover from flare tip)",
        "statutory_audit_finding": (
            "Non-destructive testing (NDT) ultrasonic thickness survey indicated localized shell thinning. "
            "Full grid mapping required during turnaround (MRPL Safety Audit 2026)."
        ),
        "referenced_documents": [
            "MRPL_MAH_Factory_Annual_Safety_Audit_2026.pdf",
            "OISD_156_Fire_Protection_Refineries.docx",
        ],
    },
    "PRV-901": {
        "tag": "PRV-901",
        "unit": "Single Point Mooring (SPM) Offshore Crude Terminal",
        "equipment_type": "Submarine Pipeline Surge Relief Valve",
        "service": "16.5 km 48-inch Offshore Crude Unloading Pipeline Protection",
        "normal_operating_range": "6.0 – 9.5 bar(g) during VLCC tanker discharge",
        "trip_limit": "Set Pressure 14.5 bar(g) (Subsea pipeline design limit 16.0 bar)",
        "failsafe_action": "Dumps crude surge volume into onshore relief tank to prevent water hammer rupture",
        "referenced_documents": [
            "MRPL_SPM_Offshore_Crude_Unloading_SOP.docx",
            "MRPL_Instrumentation_Master_Index.xlsx",
        ],
    },
    "QCV-101": {
        "tag": "QCV-101",
        "unit": "Hydrocracker Unit (HCU)",
        "equipment_type": "Inter-Bed Cold Hydrogen Quench Control Valve",
        "service": "Exotherm Control between Catalyst Beds 1 and 2",
        "normal_operating_range": "15,000 – 25,000 Nm3/h pure H2 flow",
        "trip_limit": "Auto-opens on Bed 1 delta-T > 35°C",
        "failsafe_action": "Fail Open (FO) to ensure maximum cooling in case of instrument air loss",
        "referenced_documents": [
            "MRPL_HCU_Hydrocracker_Operating_SOP.docx",
        ],
    },
}


def get_equipment_dossier(tag: str, rag_hits: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve or construct a comprehensive equipment dossier for a given tag.
    First checks the MRPL verified registry, then mines dynamic findings from RAG hits.
    """
    if not tag:
        return None

    clean_tag = tag.strip().upper()

    # 1. Check verified seed registry
    if clean_tag in MRPL_EQUIPMENT_REGISTRY:
        dossier = dict(MRPL_EQUIPMENT_REGISTRY[clean_tag])

        # Dynamically append any additional findings from RAG hits
        if rag_hits:
            extra_sources = set(dossier.get("referenced_documents", []))
            for hit in rag_hits:
                fn = hit.get("filename")
                if fn and fn not in extra_sources:
                    extra_sources.add(fn)
            dossier["referenced_documents"] = sorted(list(extra_sources))

        return dossier

    # 2. Dynamic tag miner: if not in seed registry, build from RAG hits
    if rag_hits:
        matching_snippets = []
        sources = set()
        for hit in rag_hits:
            text = hit.get("text", "")
            tags = [t.upper() for t in hit.get("tags", [])]
            if clean_tag in tags or clean_tag in text.upper():
                sources.add(hit.get("filename"))
                matching_snippets.append(text[:250])

        if matching_snippets:
            return {
                "tag": clean_tag,
                "unit": "General MRPL Refinery Infrastructure",
                "equipment_type": "Instrument / Mechanical Asset",
                "service": f"Service identified in {len(sources)} document(s)",
                "normal_operating_range": "Refer to primary SOP excerpt",
                "trip_limit": "Refer to primary SOP excerpt",
                "referenced_documents": sorted(list(sources)),
                "dynamic_notes": matching_snippets[:2],
            }

    return None


def format_dossier_box(dossier: Dict[str, Any]) -> str:
    """
    Format an equipment dossier as an industrial evidence header for the RAG prompt.
    """
    if not dossier:
        return ""

    lines = [
        "==================================================================",
        f"INDUSTRIAL ASSET DOSSIER: {dossier.get('tag')} ({dossier.get('unit', 'Refinery')})",
        f"Equipment Type: {dossier.get('equipment_type', 'Asset')}",
    ]

    if dossier.get("service"):
        lines.append(f"Service: {dossier['service']}")
    if dossier.get("normal_operating_range"):
        lines.append(f"Normal Operating: {dossier['normal_operating_range']}")
    if dossier.get("trip_limit"):
        lines.append(f"Trip / Limit: {dossier['trip_limit']}")
    if dossier.get("failsafe_action"):
        lines.append(f"Failsafe Action: {dossier['failsafe_action']}")
    if dossier.get("statutory_audit_finding"):
        lines.append(f"Audit Status: {dossier['statutory_audit_finding']}")

    docs = ", ".join(dossier.get("referenced_documents", []))
    lines.append(f"Cross-Referenced Documents: {docs}")
    lines.append("==================================================================")

    return "\n".join(lines)
