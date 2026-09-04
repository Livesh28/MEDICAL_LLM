# Phase 1: RAG V2 Architecture & Pipeline Audit

## Overview
This document audits the existing RAG pipeline, chunking format, metadata structure, BM25 retrieval baseline, and query processing in the **Medical VR – Venipuncture Training Simulation** repository.

---

## 1. Existing Retrieval Architecture Sitemap

| Component | File Path | Current Capability & Role |
| :--- | :--- | :--- |
| **TF-IDF Vector DB** | [`rag/database.py`](file:///Users/livesh/Medical_LLM/rag/database.py) | Cosine similarity over term frequency matrix (`data/rag_db`). Recall@3 = **72.0%**. |
| **BM25 Searcher** | [`scripts/run_retrieval_matrix.py`](file:///Users/livesh/Medical_LLM/scripts/run_retrieval_matrix.py) | Okapi BM25 lexical search ($k_1=1.5, b=0.75$). Recall@3 = **84.0%**, MRR = **0.820**. |
| **Dual-Engine Pipeline** | [`rag/pipeline.py`](file:///Users/livesh/Medical_LLM/rag/pipeline.py) | Formats RAG context & routes queries to Ollama `llama3.2:3b` or PyTorch fallback. |
| **Intent Router Layer** | [`api/intent_router.py`](file:///Users/livesh/Medical_LLM/api/intent_router.py) | Classifies VR intent (`NEXT_STEP`, `REPEAT`, `WHY_WRONG`, `UNSUPPORTED`) for deterministic handling. |
| **FastAPI REST API** | [`api/server.py`](file:///Users/livesh/Medical_LLM/api/server.py) | Exposes `POST /ask` endpoint with VR step context (`current_step`, `step_name`, `last_mistake`). |

---

## 2. Chunking & Metadata Structure Audit
* **Indexed Chunks Count:** 1,888 total chunks stored in [`data/rag_db/chunks.json`](file:///Users/livesh/Medical_LLM/data/rag_db/chunks.json).
* **Metadata Schema per Chunk:**
  ```json
  {
    "chunk_id": "clinical_001",
    "text": "Before performing venipuncture, hand hygiene must be performed...",
    "topic": "hand_hygiene",
    "step": 0,
    "source_id": "SRC_WHO_01",
    "source_section": "Section 2.1 Hand Hygiene Protocols",
    "source_page": "14",
    "source_url": "https://www.who.int/publications/i/item/9789241547826"
  }
  ```
* **Chunk Quality Status:** 100% unique text chunks, 0 empty chunks, 0 duplicate text records.

---

## 3. Key Limitations & RAG V2 Upgrade Targets
1. **Query Preprocessing Gaps:** Lack of controlled medical term normalization (e.g. mapping *"clean arm"* $\to$ *"skin preparation antecubital fossa"*).
2. **Metadata Under-utilization:** BM25 search does not currently leverage metadata fields (`step`, `topic`, `source_id`) as ranking signals.
3. **Lack of Post-Generation Grounding Validation:** LLM responses are not currently checked post-generation for unsupported entities or hallucinated values before returning to Unity.
