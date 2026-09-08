ByteForce-RAG (SIH 26117) - Organized Knowledge Base Structure

The repository data directory is partitioned into two primary functional categories (OCR and RAG) plus system storage for clean maintenance and modular synchronization:

================================================================================
1. data/ocr/ (OCR Subsystem Inputs & Testing)
================================================================================
Contains all scanned documents, inspection checklists, accident reports, and images
that trigger the local RapidOCR engine, PyMuPDF in-memory rasterization, and ISA-5.1
industrial tag correction.

Folders:
- 09_Inspection_Accident/
  * MRPL_MAH_Factory_Annual_Safety_Audit_2026.pdf (Scanned statutory safety audit)
  * Checklist_for_inspection_of_2-cb-MAH_Factories.pdf
  * check_list_for_preparing_accident_investigation_report.pdf
  * Inspection_Report_Format_Checklis.pdf
  * 20250618_letter.pdf
- 02_Templates_Checklists_Forms/
  * Inspection checklists, check sheets, calibration forms, and revision templates.
- scanned_samples/
  * Sample high-resolution scanned report pages (e.g. scanned_mah_page.png).

Sync CLI:
  python -m app.ingest ocr

================================================================================
2. data/rag/ (Structured RAG Subsystem Knowledge Base)
================================================================================
Contains all digital refinery procedures, engineering standards, operating manuals,
instrumentation index spreadsheets, and CAD drawing assets that form the core RAG
retrieval knowledge base.

Folders:
- 01_Standards_Reference/
  * MRPL Operating SOPs (HCU, FCCU, CDU/VDU, SPM Offshore Unloading)
  * MRPL_Refinery_Master_Technical_Profile.docx (15.0 MMTPA complex profile)
  * OISD-156 Fire Protection Standards
  * Instrumentation and Control Specifications
- 03_PIDs/
  * Standard P&ID legends and process flow CAD drawings (.dwg, .dxf)
- 04_Instrumentation_Drawings/
  * MRPL_Instrumentation_Master_Index.xlsx (Master equipment & instrument index)
  * Standard instrument mounting and transmitter detail drawings
- 05_Control_Panel_Drawings/
  * ICP layouts, BOMs, wiring schematics, power distribution
- 07_Network_Architecture/
  * Sample network topology and architecture drawings
- 08_SCADA/
  * SCADA hardware specifications and screen layouts
- 10_Public_Reference/
  * Public reference specifications and documentation

Sync CLI:
  python -m app.ingest rag

================================================================================
3. System State, Indexes & Storage (Auto-managed)
================================================================================
- qdrant_storage/       : Local on-disk embedded Qdrant vector database
- document_registry.db  : SQLite persistent registry tracking document hashes and sync status
- bm25_index.json       : Okapi BM25 inverted index for exact equipment tag keyword retrieval
- airgap_audit.log      : Sovereign append-only network audit trail
- file_manifest.csv     : Master manifest with category and retrieval mode mapping

================================================================================
General Sync Command:
  python -m app.ingest      # Incrementally syncs both OCR and RAG documents
================================================================================
