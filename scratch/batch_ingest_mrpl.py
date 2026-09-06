"""
Batch ingest the newly generated MRPL refinery documents.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ingestion import IngestionEngine
DATA_DIR = BASE_DIR / "data"

new_files = [
    DATA_DIR / "01_Standards_Reference" / "MRPL_FCCU_Operating_Manual.docx",
    DATA_DIR / "01_Standards_Reference" / "MRPL_CDU_VDU_Crude_Distillation_SOP.docx",
    DATA_DIR / "01_Standards_Reference" / "OISD_156_Fire_Protection_Refineries.docx",
    DATA_DIR / "02_Templates_Checklists_Forms" / "MRPL_Refinery_Turnaround_Shutdown_SOP.docx",
    DATA_DIR / "04_Instrumentation_Drawings" / "MRPL_Instrumentation_Master_Index.xlsx",
]

engine = IngestionEngine()
for f in new_files:
    if f.exists():
        print(f"Ingesting: {f.name}...")
        res = engine.ingest_document(f)
        print(f"-> {res}")

engine.close()
print("Batch ingestion of new MRPL documents complete!")
