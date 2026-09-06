"""
Batch ingest newly created authentic MRPL documents.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ingestion import IngestionEngine

files = [
    BASE_DIR / "data" / "01_Standards_Reference" / "MRPL_Refinery_Master_Technical_Profile.docx",
    BASE_DIR / "data" / "01_Standards_Reference" / "MRPL_HCU_Hydrocracker_Operating_SOP.docx",
    BASE_DIR / "data" / "01_Standards_Reference" / "MRPL_Crude_Assay_and_Blend_Optimization.xlsx",
    BASE_DIR / "data" / "01_Standards_Reference" / "MRPL_SPM_Offshore_Crude_Unloading_SOP.docx",
    BASE_DIR / "data" / "01_Standards_Reference" / "MRPL_KSPCB_Environmental_Compliance_Report.docx",
]

engine = IngestionEngine()
for f in files:
    if f.exists():
        print(f"Ingesting: {f.name}...")
        res = engine.ingest_document(f)
        print(f"-> Result: {res}")

engine.close()
print("All authentic MRPL documents ingested successfully!")
