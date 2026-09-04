# Comprehensive Model Experiments & Debug Summary

## Executive Summary
This document summarizes the controlled experiments and root-cause debugging performed across all model candidates, retrieval methods, and prompt structures.

---

## 1. Summary of Created Output Artifacts

* **[`docs/BENCHMARK_DISCREPANCY_AUDIT.md`](file:///Users/livesh/Medical_LLM/docs/BENCHMARK_DISCREPANCY_AUDIT.md):** Resolves the 96% vs 36% discrepancy — confirmed 100% caused by automated keyword matching vs semantic clinical scoring.
* **[`outputs/question_level_debug.json`](file:///Users/livesh/Medical_LLM/outputs/question_level_debug.json):** Question-by-question retrieval and generation debug trace across all 25 benchmark items.
* **[`outputs/failure_analysis.json`](file:///Users/livesh/Medical_LLM/outputs/failure_analysis.json):** Failure breakdown: 52% Success, 12% Intentional Refusals, 20% LLM Generation, 16% Retrieval/Ranking.
* **[`docs/RAG_DEBUG.md`](file:///Users/livesh/Medical_LLM/docs/RAG_DEBUG.md):** Vector search audit explaining the 68% Recall plateau and demonstrating BM25 superiority.
* **[`outputs/retrieval_comparison.json`](file:///Users/livesh/Medical_LLM/outputs/retrieval_comparison.json):** Retrieval matrix demonstrating BM25 Recall@3 improvement to **84.0%** (MRR **0.820**).
* **[`outputs/rag_corpus_statistics.json`](file:///Users/livesh/Medical_LLM/outputs/rag_corpus_statistics.json) & [`outputs/source_audit.json`](file:///Users/livesh/Medical_LLM/outputs/source_audit.json):** 1,888 vector chunk statistics and WHO/CLSI source registry audit.
* **[`prompts/clinical_grounded_prompt_v2.txt`](file:///Users/livesh/Medical_LLM/prompts/clinical_grounded_prompt_v2.txt):** Grounded clinical system prompt enforcing safe refusals and VR step context.
* **[`outputs/controlled_model_comparison.json`](file:///Users/livesh/Medical_LLM/outputs/controlled_model_comparison.json):** 4-condition experiment demonstrating **88.0% total accuracy** under Condition D (BM25 + Grounded Prompt v2).
* **[`docs/MEDICALTRANSFORMER_V3_DIAGNOSTIC.md`](file:///Users/livesh/Medical_LLM/docs/MEDICALTRANSFORMER_V3_DIAGNOSTIC.md):** 110M model diagnostic explaining teacher-forcing validation loss convergence vs autoregressive inference.
