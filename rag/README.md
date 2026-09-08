# Sovereign Industrial RAG Engine (SIH 26117)

**Subsystem Owner**: Harsha Bacham (RAG Pipeline & Retrieval Architecture)  
**Problem Statement**: Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for Confidential Industrial Work (MRPL)

---

## Architecture Overview

The RAG Engine provides **100% sovereign, on-premise, air-gapped retrieval** over confidential refinery SOPs, standards, inspection reports, equipment master sheets, and CAD schematics.

```
User Query (e.g. "What is the trip limit for PT-101?")
                       │
                       ▼
Refinery Domain Query Expansion (Offline MRPL Ontology: CDU, HCU, FCCU, EDS, SIL-3)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
Dense Semantic Stream           Sparse Lexical Stream
(BAAI/bge-m3, 1024-dim,         (Okapi BM25 Index, exact
 Qdrant Vector Store)            alphanumeric tag preservation)
       │                               │
       └───────────────┬───────────────┘
                       │
                       ▼
       Reciprocal Rank Fusion (RRF, k=60)
                       │
                       ▼
       Similarity Threshold Cutoff (score >= 0.35) & Page Diversity Filter
                       │
                       ▼
       Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)
                       │
                       ▼
       Equipment Knowledge Graph (Instant Master Dossier Box for XV-301, BDV-701)
       Numerical Grounding Verification (Safety limits, stroke times, bar(g), °C)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
Local LLM Grounded Answer       Structured Agent JSON Interface
(Ollama: Qwen2.5-7B)            (rag.search() + Qwen2.5-VL Vision Handoffs)
```

---

## Key Features

1. **Dual-Stream Hybrid Retrieval**:
   Dense embeddings (`bge-m3`) capture semantic context; Okapi BM25 captures exact equipment tags (`PT-101`, `XV-301`). Fused via Reciprocal Rank Fusion.
2. **Cross-Encoder Deep Attention**:
   `bge-reranker-v2-m3` scores query-document pairs simultaneously, ensuring technical specifications rank #1.
3. **Hierarchical Semantic Chunking**:
   Splits along Markdown headers $\rightarrow$ paragraphs $\rightarrow$ sentence boundaries. Retains parent-child procedural context (`parent_id`, `parent_text`).
4. **Relational Equipment Graph**:
   Injects verified technical dossiers (trip limits, failsafe action, statutory audit findings) for major MRPL equipment.
5. **Anti-Hallucination Guardrails**:
   Strict similarity cutoff ($0.35$), numerical grounding verification, and safe fallback response:
   *"The local knowledge base does not contain enough relevant information to answer this."*
6. **Agent JSON Contract**:
   `rag.search()` provides structured payloads with confidence metrics and CAD visual handoff descriptors for Bhuvan's Vision module (`Qwen2.5-VL`).

---

## Python API Usage

```python
from rag import LocalRAG, retrieve, build_context

# 1. Direct retrieval
hits = retrieve("What is the stroke time limit for XV-301?", top_k=3)
context, sources = build_context(hits)

# 2. Agent JSON Interface
rag = LocalRAG()
result = rag.search("What are the key findings in the safety audit?", top_k=3)
print(f"Found: {result['found']}, Confidence: {result['confidence']}")
for source in result['sources']:
    print(f"- {source['filename']} (Page {source['page_number']}, Score: {source['score']})")

# 3. Grounded Answer Generation
answer, sources = rag.answer("What is the trip limit for PT-101?")
print(f"Answer: {answer}")
```

---

## CLI Usage

```powershell
# Run self-test or query via RAG CLI
python -m rag "What is the trip limit for PT-101?"
```
