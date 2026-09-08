# ByteForce-RAG: Sovereign On-Premise Industrial RAG & Local OCR Subsystem

**SIH Problem Statement SIH26117**  
*Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for Confidential Industrial Work*

**Subsystem Owner**: Harsha Bacham (RAG Enhancement & Document Understanding Pipeline)

---

## Architecture Overview

ByteForce-RAG is a **100% sovereign, local, and air-gapped** document ingestion, OCR, and retrieval engine designed for confidential industrial documents (engineering reports, inspection checklists, P&IDs, standards, manuals, and drawings).

```
Industrial Document (PDF, Images, DOCX, XLSX)
                 │
                 ▼
       Format Identification
                 │
   ┌─────────────┴─────────────┐
   ▼                           ▼
Digital PDF / DOCX / XLSX   Scanned PDF Page / Image
(Extract Native Text)       (Render in-memory RGB Image)
   │                           │
   │ Text Quality Check        ▼
   └────────► (Insufficient) ──► Local RapidOCR (ONNX Runtime)
                               │ (Extract text + confidences)
                               ▼
               Structured Records & Metadata
             (source, page_number, location, tags,
              extraction_method, ocr_used, ocr_confidence)
                               │
                               ▼
                    Semantic Chunker
           (Paragraph/Sentence-boundary splitting,
            Deterministic Chunk IDs: p1_c0, p1_c1...)
                               │
                               ▼
                   EmbeddingProvider
                 (BAAI/bge-m3, 1024-dim)
                               │
                               ▼
                 Qdrant Vector Store
           (Server URL or Embedded Local Disk)
                               │
                               ▼
                    SQLite Registry
               (Hash, Status, OCR stats)
```

---

## Key Features

1. **100% Air-Gapped Local OCR**:
   - Automated detection of scanned vs native text pages using text quality heuristics.
   - High-resolution in-memory PDF page rendering via PyMuPDF (no external Poppler binary).
   - Local OCR via `RapidOCR` running on `onnxruntime` with zero cloud calls.
   - Preserves technical industrial strings (`PT-101`, `FT-204`, `XV-301`, `24 VDC`, `120 VAC`).

2. **Multi-Format Ingestion**:
   - **PDF**: Automatic hybrid handling (native digital text pages extracted directly; image/scanned pages rendered and OCR'd).
   - **Images** (`.png`, `.jpg`, `.jpeg`, `.webp`): Direct local OCR extraction.
   - **DOCX**: Extracts both textual paragraphs and structured table blocks.
   - **XLSX**: Extracts worksheets with header association and row structures.

3. **Semantic Chunking & Provenance Retention**:
   - Splits on natural paragraph and sentence boundaries (no mid-word or mid-sentence cuts).
   - Stable, collision-free chunk IDs (`p{page}_c{offset}`) preventing vector overwrite bugs.
   - Full provenance payload stored in Qdrant: `document_id`, `chunk_id`, `text`, `source`, `filename`, `location`, `page_number`, `category`, `document_type`, `tags`, `ocr_used`, `extraction_method`, `section`.

4. **Modular Embedding Provider**:
   - `LocalEmbeddingProvider`: In-process `BAAI/bge-m3` (1024 dimensions).
   - `APIEmbeddingProvider`: HTTP client interface ready for Nandan's future model laptop handoff.

5. **Standalone Embedded Storage Fallback**:
   - Attempts connection to Qdrant server (`http://localhost:6333`).
   - If Docker is offline or not installed, automatically falls back to embedded local disk storage at `data/qdrant_storage`.

6. **Anti-Hallucination Safe Retrieval**:
   - Strict similarity threshold filtering (`SIMILARITY_THRESHOLD = 0.35`).
   - Safe fallback when knowledge is missing:
     *"The local knowledge base does not contain enough relevant information to answer this."*
   - SIH citation formatting exposing document name, exact page number, extraction method, and equipment tags.

---

## Clean Python APIs for Team Integration

Other workbench modules (Qwen3 Agent Loop, Vision Pipeline, Frontend API) can interface directly:

```python
from pathlib import Path

# 1. Ingest a document
from app.ingestion import ingest_document
result = ingest_document(Path("data/09_Inspection_Accident/report.pdf"))
print(result)  # {'status': 'indexed', 'chunk_count': 6, 'ocr_used': True}

# 2. Local OCR text extraction
from app.ocr import extract_text_with_ocr
text, confidence = extract_text_with_ocr("data/drawing.png")

# 3. Retrieve relevant evidence chunks
from app.rag import retrieve, build_context
hits = retrieve("What was the corrosion status of PT-101?", top_k=3)
context, sources = build_context(hits)

# 4. Generate grounded answer
from app.rag import LocalRAG
rag = LocalRAG()
answer, sources = rag.answer("What were the major findings?")
```

---

## CLI Commands

### 1. Run System Health Checks
```powershell
python -m app.check
```
Verifies status of Qdrant (server + embedded disk), Ollama, Local OCR engine, and BGE-M3 embedding provider.

### 2. Ingest All Documents (Batch Sync)
```powershell
python -m app.ingest
```
Performs incremental sync: skips unchanged files, re-embeds modified files, cleans deleted files, and prints an 8-metric summary.

### 3. Ingest Single Document
```powershell
python -m app.ingest "data/09_Inspection_Accident/check_list_for_preparing_accident_investigation_report.pdf"
```

### 4. Run Streamlit UI
```powershell
streamlit run app/ui.py
```

### 5. Run Complete Automated Test Suite
```powershell
python -m unittest discover -s tests
```

---

## Configuration Settings (`app/config.py`)

| Setting | Default | Purpose |
| :--- | :--- | :--- |
| `QDRANT_URL` | `http://localhost:6333` | Server URL for Qdrant |
| `QDRANT_STORAGE_PATH` | `data/qdrant_storage` | Local on-disk path for embedded Qdrant fallback |
| `QDRANT_PREFER_LOCAL` | `False` | When True or if server is unreachable, use embedded disk |
| `OCR_ENABLED` | `True` | Master toggle for local OCR pipeline |
| `OCR_ENGINE` | `"rapidocr"` | Local engine (`rapidocr`, `tesseract`, `mock`, `auto`) |
| `OCR_MIN_TEXT_LENGTH` | `50` | Character threshold below which pages trigger OCR |
| `OCR_CONFIDENCE_THRESHOLD` | `0.5` | Minimum line confidence to keep |
| `OCR_DPI` | `200` | Resolution for rendering PDF pages to image |
| `EMBEDDING_MODEL` | `"BAAI/bge-m3"` | Local embedding model name |
| `EMBEDDING_PROVIDER` | `"local"` | `"local"` (in-process) or `"api"` (remote HTTP) |
| `EMBEDDING_API_URL` | `http://localhost:8000/embed` | Target URL when `EMBEDDING_PROVIDER = "api"` |
| `CHUNK_SIZE` | `900` | Target character size per semantic chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between adjacent chunks |
| `TOP_K` | `5` | Number of chunks retrieved for context |
| `SIMILARITY_THRESHOLD` | `0.35` | Minimum cosine similarity score threshold |
| `OLLAMA_MODEL` | `"qwen3:8b"` | Target local LLM for grounded answering |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Local Ollama chat API endpoint |

---

## Verification & Test Summary

All 26 automated unit and integration tests are passing:
- `tests/test_ocr.py`: Scanned page insufficiency check, technical tag preservation, mock engine.
- `tests/test_ocr_integration.py`: Real RapidOCR extraction, PyMuPDF rendering, hybrid PDF page-by-page extraction.
- `tests/test_parsers.py`: PDF, DOCX tables/paragraphs, XLSX sheets/rows, Image OCR, bad format handling.
- `tests/test_chunking.py`: Semantic paragraph splitting, collision-free chunk IDs, metadata retention.
- `tests/test_registry.py`: SQLite incremental sync, skip unchanged, update modified, delete removed.
- `tests/test_store.py`: Qdrant payload serialization, metadata filtering, point ID uniqueness.
- `tests/test_rag.py`: Threshold filtering, citation provenance formatting, safe fallback message.
- `tests/test_e2e_demo.py`: Complete SIH simulated scanned report workflow.


## SIH26117 Agent + RAG Integration

This merged build keeps the Agent package from the Agent Workbench and the
enhanced ByteForce-RAG subsystem as separate layers.

- `app/agent/agent.py` = orchestration, classification, planning, verification and replanning.
- `app/agent/planner.py` = execution plan and bounded dynamic replanning.
- `app/agent/verifier.py` = local evidence verification gate.
- `app/rag.py` = hybrid BGE-M3 + BM25 + RRF + reranker retrieval.
- `app/ingestion.py` = file -> parser/OCR -> chunks -> embeddings -> Qdrant + BM25.
- `data/` = original source documents and local indexes/registry.

Run:
```powershell
docker compose up -d
python -m app.ingest
streamlit run test_agent.py
```

For a clean migration from an older index, recreate the Qdrant collection and
remove the old BM25/registry state before running ingestion again.
