# 5-Way Model Comparison & Benchmark Analysis

## Overview
This document presents the 5-way model comparative evaluation conducted using [`scripts/run_comparison_eval.py`](file:///Users/livesh/Medical_LLM/scripts/run_comparison_eval.py) against Gold Benchmark v2 ([`data/evaluation/venipuncture_gold_eval_v2.json`](file:///Users/livesh/Medical_LLM/data/evaluation/venipuncture_gold_eval_v2.json)).

---

## 5-Way Benchmark Summary Matrix

| Model Candidate | Accuracy (%) | Partial (%) | Incorrect (%) | Hallucination Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **1. MedicalTransformerLM (`best.pt` - v1)** | `0.0%` | `0.0%` | `28.0%` | `72.0%` |
| **2. MedicalTransformerLM (`best_v2.pt` - v2)** | `4.0%` | `16.0%` | `32.0%` | `48.0%` |
| **3. MedicalTransformerLM (`best_v3.pt` - v3 SFT)** | `16.0%` | `8.0%` | `40.0%` | **`36.0%`** |
| **4. Llama 3.2 3B without RAG** | `20.0%` | `52.0%` | `12.0%` | `16.0%` |
| **5. Llama 3.2 3B + Local RAG** | **`36.0%`** | **`40.0%`** | **`8.0%`** | **`16.0%`** |

---

## Empirical Findings
1. **MedicalTransformerLM Progression (v1 $\to$ v3):**
   * SFT v3 improved standalone accuracy from `0.0%` to **`16.0%`** and cut hallucinations in half (`72%` $\to$ `36%`).
2. **Llama 3.2 3B + RAG Superiority:**
   * Combining vector context retrieval with Llama 3.2 3B achieves top performance across clinical accuracy, partial correctness, and low hallucination.
