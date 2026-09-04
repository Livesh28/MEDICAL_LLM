# Phase 10: MedicalTransformerLM v3 Diagnostic Report

## Executive Summary
This diagnostic investigates why `MedicalTransformerLM` (110.04M PyTorch model) achieved a low validation loss (**`0.1806`**, PPL `1.20`) during SFT v3 training, yet only scored **16.0% accuracy** on the unconstrained benchmark.

---

## 1. Diagnostic Findings

### A. Overfitting & Teacher Forcing Discrepancy
* During training, loss is calculated with teacher-forced target tokens where the preceding ground truth context is always provided.
* During inference, the model generates tokens autoregressively. Once a small token deviation occurs, errors cascade, causing the 110M model to emit repetitive subwords.

### B. Parameter Capacity Limit
* At ~110M parameters ($D=768, L=12, H=12$), the model has insufficient capacity to retain precise verbatim phlebotomy facts (such as tube inversion counts and exact CLSI order of draw) without RAG evidence retrieval.

### C. Improvement from SFT v1 to SFT v3
* **v1 Base (`best.pt`):** Accuracy `0.0%`, Hallucinations `72.0%`
* **v2 SFT (`best_v2.pt`):** Accuracy `4.0%`, Hallucinations `48.0%`
* **v3 SFT (`best_v3.pt`):** Accuracy **`16.0%`**, Hallucinations **`36.0%`** (4x accuracy gain and 50% hallucination reduction!).

---

## 2. Recommendation
Retain `checkpoints/best_v3.pt` as the primary offline research candidate. Do not retrain `best_v4.pt` until data scaling is explicitly instructed.
