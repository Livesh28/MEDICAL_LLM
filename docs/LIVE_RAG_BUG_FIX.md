# Live RAG Bug Fix & Pipeline Optimization Report

## Executive Summary
This document summarizes the root-cause diagnosis, architectural fixes, empirical debug logs, and regression test results for the **Medical Vector RAG Assistant Bug Fix Phase**.

The previous UI failure (e.g. asking *"What is hypertension"* returning a chunk about scleroderma, `Chunk ID: undefined`, and malformed answer `"What is the body? <|endoftext|>"`) has been fully diagnosed and resolved across all pipeline layers.

---

## 1. Root Cause Analysis

| Problem Area | Root Cause | Architectural Fix |
| :--- | :--- | :--- |
| **Problem 1: Retrieval Relevance** | `split()` in BM25 tokenization retained trailing punctuation (`"hypertension?" != "hypertension"`). Common stopwords ("what", "is") matched unrelated chunks. `TYPO_SYNONYMS` injected broad terms ("high blood pressure") into vector matching, diluting precision. | Implemented strict stopword filtering, regex alphanumeric tokenization in `rag/retriever_v2.py`, zero-keyword metadata boost suppression, and cleaned `TYPO_SYNONYMS` in `rag/database.py`. |
| **Problem 2: Source Metadata (`Chunk ID: undefined`)** | `sources` payload in `rag/pipeline.py` did not explicitly pass `chunk_id`, causing frontend JavaScript `${s.chunk_id}` to evaluate to `undefined`. | Updated `rag/pipeline.py` to always pass `chunk_id` along with `source_id`, `topic`, `step`, `score`, `url`, and `snippet`. Added fallback rendering in `api/server.py`. |
| **Problem 3 & 4: Generation & Special Token Leakage** | Fallback generator did not receive RAG context, and tokenizer output was not sanitized, allowing `<|endoftext|>` and `Instruction:` template headers to leak into UI. | Added `sanitize_response_text()` function in `rag/pipeline.py` and `api/server.py`. Formatted fallback prompts with evidence context. |
| **Problem 5: Model Engine Routing** | Runtime response did not consistently log active engine details. | Updated `rag/pipeline.py` to explicitly return `"engine": "llama3.2:3b"`, `"engine": "pytorch_medical_transformer_v3"`, `"engine": "vr_safety_guardrail"`, or `"engine": "vr_stepmanager_deterministic"`. |
| **Problem 6 & 7: Relevance Threshold & Refusal** | Low-relevance out-of-domain queries attempted to generate answers from irrelevant evidence. | Implemented relevance score thresholding (`relevance_threshold = 1.0`). Below threshold, returns safe refusal: *"I don't have enough verified information in the current knowledge base to answer that reliably."* |
| **Problem 8: Prompt Construction** | Prompt structure lacked explicit section separation between evidence, VR context, and question. | Updated Grounded Prompt v3 format cleanly separating `SYSTEM:`, `EVIDENCE:`, `VR CONTEXT:`, `QUESTION:`, `RULES:`, and `ANSWER:`. |
| **Problem 9: BM25 Verification** | Unverified document scoring logic. | Created `test_01_bm25_ranking_document_a_vs_document_b` unit test verifying Document A ("Hypertension") ranks #1 for "What is hypertension?" and Document B ("Scleroderma") ranks #1 for "What is scleroderma?". |
| **Problem 12: VR Deterministic Routing** | Deterministic VR queries (`NEXT_STEP`, `WHY_WRONG`) risk LLM hallucination. | Intent router directs `NEXT_STEP`, `REPEAT`, `WHY_WRONG` to deterministic `StepManager` state in `api/intent_router.py`. |

---

## 2. Debug Artifacts Generated

1. **`outputs/live_query_retrieval_debug.json`**:
   Contains step-by-step query normalization, query tokens, top 5 retrieved chunks, scores, topics, and relevance determination for 7 benchmark queries:
   - What is hypertension?
   - What is diabetes?
   - What is venipuncture?
   - Why is the venipuncture site cleaned?
   - What is a tourniquet?
   - What is blood collection?
   - What should I do at step 11?

2. **`outputs/live_generation_debug.json`**:
   Contains inference requests, raw output, sanitized output, token leakage verification, engine logs, and latency measurements.

---

## 3. Empirical Test Results (`scripts/test_live_rag.py`)

```
.......
----------------------------------------------------------------------
Ran 7 tests in 18.451s

OK
```

- **Test 01 (BM25 Document A vs B)**: PASSED
- **Test 02 (Valid `chunk_id` Metadata)**: PASSED
- **Test 03 (Special Token Sanitization)**: PASSED
- **Test 04 (Unsupported Question Safe Refusal)**: PASSED
- **Test 05 (VR Deterministic Routing)**: PASSED
- **Test 06 (Model Engine Logging)**: PASSED
- **Test 07 (Supported Medical Question)**: PASSED

---

## 4. Before & After Pipeline Comparison

### Before Fix (Observed UI Failure):
- **User Query**: `"What is hypertension"`
- **Retrieved Chunk**: `"What does screening for scleroderma entail?"`
- **Output Answer**: `"What is the body? <|endoftext|>"`
- **Metadata**: `Chunk ID: undefined`, Score: `7.5308`

### After Fix (Empirical Verification):
- **User Query**: `"What is hypertension"`
- **Retrieved Chunk #1**: `doc_12_chunk_0` (*"Hypertension, commonly referred to as high blood pressure, is a chronic medical condition..."*)
- **Output Answer**: `"Hypertension is a chronic medical condition characterized by elevated blood pressure."`
- **Metadata**: `Chunk ID: doc_12_chunk_0`, Score: `6.5334`, Engine: `llama3.2:3b`
- **Special Tokens**: 0 leaked (`<|endoftext|>` completely stripped)
