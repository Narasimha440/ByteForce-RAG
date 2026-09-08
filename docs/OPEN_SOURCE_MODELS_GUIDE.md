# Sovereign On-Premise Open-Weight Models Guide (SIH 26117)

**Problem Statement SIH26117**: Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for Confidential Industrial Work (Sponsored by **Mangalore Refinery and Petrochemicals Limited - MRPL**)

This guide provides a comprehensive evaluation of open-weight foundation models for 100% air-gapped, on-premise execution on local enterprise GPUs and gaming laptops (8GB – 16GB VRAM).

---

## 1. Role-by-Role Model Architecture Matrix

| Subsystem / Team Member | Primary Recommended Model | Alternative / Fallback | Quantization | Min VRAM | Primary Capabilities |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RAG Retrieval & Embeddings** (Harsha) | **`BAAI/bge-m3`** | `intfloat/multilingual-e5-large` | FP16 / ONNX | ~2.2 GB | 1024-dim, 8192 context window, dense + sparse representations, multi-lingual. |
| **Cross-Encoder Reranker** (Harsha) | **`BAAI/bge-reranker-v2-m3`** | `ms-marco-MiniLM-L-6-v2` | FP16 / INT8 | ~1.1 GB | Joint query-document cross-attention, boosting technical tag & equipment code accuracy. |
| **Document OCR Engine** (Harsha) | **`RapidOCR`** (ONNX Runtime) | `PyMuPDF` + `Tesseract` | ONNX INT8 | ~0.5 GB | Pure-local, zero-cloud C++ binary requirement; detects scanned vs native PDF pages. |
| **Multimodal Vision (VL)** (Bhuvan) | **`Qwen2.5-VL-7B-Instruct`** | `InternVL2.5-8B` | Q4_K_M / Q5_K_M | ~6–8 GB | State-of-the-art vision model for reading P&IDs, instrument detail CAD exports, single-line diagrams, and charts. |
| **General Agent & Planning** (Team) | **`Qwen2.5-7B-Instruct`** or **`Qwen2.5-14B-Instruct`** | `Mistral-7B-Instruct-v0.3` | Q4_K_M / Q8 | ~6–9 GB | Exceptional structured JSON, complex tool-calling, engineering instruction following. |
| **Deep Reasoning & Root Cause Analysis** | **`DeepSeek-R1-Distill-Qwen-8B`** | `DeepSeek-R1-Distill-Qwen-14B` | Q4_K_M | ~6–8 GB | Extended Chain-of-Thought (CoT) reasoning for accident investigations and major hazard failure analysis. |
| **Internal Tool & Engineering Code** | **`Qwen2.5-Coder-7B-Instruct`** | `Qwen2.5-Coder-14B-Instruct` | Q4_K_M | ~6–8 GB | Automated Python engineering calculation scripts, thermodynamic equilibrium, flow calculations. |

---

## 2. In-Depth Evaluation of Recommended Models

### A. General Agentic LLM: `Qwen2.5-7B-Instruct` / `Qwen2.5-14B-Instruct`
- **Why it fits SIH26117**:
  - Outperforms older models in technical document understanding, mathematical calculations, and strict JSON output formatting.
  - Native 128k context support (can ingest entire operating procedures or multiple P&ID data tables simultaneously).
  - Can be easily pulled and served via Ollama:
    ```powershell
    ollama pull qwen2.5:7b-instruct
    ```

### B. Root Cause Failure Analysis: `DeepSeek-R1-Distill-Qwen-8B`
- **Why it fits SIH26117**:
  - Distilled from DeepSeek-R1 using Qwen2.5 architecture, specializing in rigorous step-by-step reasoning.
  - Ideal for SIH demonstration scenarios involving Major Accident Hazard (MAH) audits: determining why a slide valve stroke time degraded or calculating corrosion rates per API 570.
  - Ollama command:
    ```powershell
    ollama pull deepseek-r1:8b
    ```

### C. Multimodal P&ID & Diagram Vision: `Qwen2.5-VL-7B-Instruct`
- **Why it fits Bhuvan's Vision Module**:
  - Specifically trained on engineering schematics, technical flowcharts, and high-resolution document images.
  - Capable of recognizing instrument bubbles (e.g. `PT`, `FT`, `XV`, `LT`), signal line types (pneumatic vs electrical vs capillary), and equipment boundaries directly from raster images.
  - Runs with llama.cpp or HuggingFace Transformers locally.

### D. Embeddings & Reranking: `BAAI/bge-m3` & `bge-reranker-v2-m3`
- **Implemented in ByteForce-RAG (`app/embeddings.py`, `app/reranker.py`)**:
  - `bge-m3` produces 1024-dim dense embeddings with high discriminative capability across technical numbers, hyphens, and units.
  - `bge-reranker-v2-m3` performs deep cross-attention over query-document pairs, boosting exact equipment tag matches and critical safety limits to rank #1.

### E. Document OCR & Tag Sanitizer: `RapidOCR` + `ISA-5.1 Sanitizer`
- **Implemented in ByteForce-RAG (`app/ocr.py`)**:
  - **Zero-Cloud, Zero-Binary Requirement**: Runs on local ONNX Runtime directly in Python without needing external Poppler or C++ Tesseract binaries.
  - **Dynamic Scanned Detection**: Heuristically determines if a PDF page lacks digital text and automatically triggers in-memory rasterization at 200 DPI via PyMuPDF.
  - **ISA-5.1 Optical Tag Correction**: Post-processes OCR text with deterministic regex rules to fix character misreads in instrumentation codes (e.g. `PT-l0l` -> `PT-101`, `24 VOC` -> `24 VDC`).
  - **Provenance Guarantee**: Tags every extracted chunk with `ocr_used = True`, `ocr_confidence`, and exact `page_number`.

---

## 3. Hardware Deployment Profiles for On-Premise Laptops

### Profile 1: Single Gaming Laptop (16GB RAM + 8GB VRAM RTX 3060 / 4060)
- **Local RAG + OCR** (Harsha): RapidOCR (CPU) + BGE-M3 (2.2GB VRAM) + Embedded Qdrant (Disk).
- **Local Agent LLM**: `qwen2.5:7b-instruct-q4_K_M` (4.5GB VRAM) served via Ollama.
- **Concurrent VRAM Usage**: ~6.7 GB VRAM (Fits comfortably within 8GB).

### Profile 2: Multi-Laptop Distributed Workbench (Team Setup)
- **Laptop 1 (Harsha - RAG & OCR)**:
  - Streamlit UI (`app/ui.py`)
  - Embedded Qdrant & SQLite Registry
  - In-process RapidOCR + Hybrid BM25
  - BGE-M3 local embedding provider
- **Laptop 2 (Nandan - Model Server)**:
  - Serves `Qwen2.5-7B` / `DeepSeek-R1:8B` on local network (`http://192.168.x.x:11434`).
  - ByteForce-RAG connects cleanly by setting `OLLAMA_URL` and `EMBEDDING_API_URL` in `app/config.py`.
- **Laptop 3 (Bhuvan - Multimodal Vision)**:
  - Serves `Qwen2.5-VL-7B` for CAD & P&ID diagram raster analysis.

---

## 4. Setting up Ollama for Local Air-Gapped Operation

```powershell
# 1. Pull the primary models (one-time setup before taking machine air-gapped)
ollama pull qwen2.5:7b-instruct
ollama pull deepseek-r1:8b

# 2. Verify installed models
ollama list

# 3. Test running a technical industrial prompt
ollama run qwen2.5:7b-instruct "What does a SIL-3 rating imply for emergency shutdown valve stroke times?"
```
