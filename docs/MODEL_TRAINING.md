# Supervised Fine-Tuning (SFT v3) Training Report

## Research Track: MedicalTransformerLM v3

This document details the Supervised Fine-Tuning (v3 research track) performed on `MedicalTransformerLM` (110.04M parameters).

---

## 1. Training Setup & Hyperparameters
* **Script:** [`training/train_sft_v3.py`](file:///Users/livesh/Medical_LLM/training/train_sft_v3.py)
* **DataLoader:** [`dataset/sft_dataloader_v3.py`](file:///Users/livesh/Medical_LLM/dataset/sft_dataloader_v3.py)
* **Loss Function:** `nn.CrossEntropyLoss(ignore_index=-100)` with **Instruction Loss Masking** (`-100` mask over prompt tokens).
* **Base Checkpoint:** [`checkpoints/best_v2.pt`](file:///Users/livesh/Medical_LLM/checkpoints/best_v2.pt)
* **Output Checkpoint:** [`checkpoints/best_v3.pt`](file:///Users/livesh/Medical_LLM/checkpoints/best_v3.pt)
* **Execution Device:** Apple Silicon MPS GPU (`mps`)
* **Epochs / Steps:** 10 Epochs / 230 Steps

---

## 2. Loss & Perplexity Convergence Log

```text
Epoch  1/10 | Step  23 | Train Loss: 5.0679 | Val Loss: 3.4242 | PPL: 30.70
Epoch  2/10 | Step  46 | Train Loss: 2.4776 | Val Loss: 2.0818 | PPL: 8.02
Epoch  3/10 | Step  69 | Train Loss: 1.1709 | Val Loss: 1.2332 | PPL: 3.43
Epoch  4/10 | Step  92 | Train Loss: 0.5498 | Val Loss: 0.6454 | PPL: 1.91
Epoch  5/10 | Step 115 | Train Loss: 0.2460 | Val Loss: 0.4788 | PPL: 1.61
Epoch  6/10 | Step 138 | Train Loss: 0.1339 | Val Loss: 0.2983 | PPL: 1.35
Epoch  7/10 | Step 161 | Train Loss: 0.0695 | Val Loss: 0.2151 | PPL: 1.24
Epoch  8/10 | Step 184 | Train Loss: 0.0349 | Val Loss: 0.1979 | PPL: 1.22
Epoch  9/10 | Step 207 | Train Loss: 0.0257 | Val Loss: 0.1899 | PPL: 1.21
Epoch 10/10 | Step 230 | Train Loss: 0.0206 | Val Loss: 0.1806 | PPL: 1.20
```

* **Outcome:** Validation loss dropped from `3.4242` to **`0.1806`**, reducing perplexity from `30.70` to **`1.20`**.
