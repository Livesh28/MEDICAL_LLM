# Multi-Model Architecture & Routing System

## Overview

The **AI-Enhanced VR Venipuncture Training System** features a multi-model architecture supporting dynamic routing between high-capacity clinical domain models, lightweight fallback models, and a preserved PyTorch research model.

```text
                        UNITY VR / WEB DASHBOARD
                                   │
                             Whisper STT
                                   │
                             Intent Router
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
       VR Query               Clinical Query            Technical QA
          │                        │                        │
     StepManager              RAG Engine               Technical RAG
          │                        │                        │
   (Deterministic)            Model Router                  │
          │                        │                        │
          │             ┌──────────┼──────────┐             │
          │             │          │          │             │
          │             ▼          ▼          ▼             │
          │        OpenBioLLM   Llama 3.2   Medical         │
          │            8B          3B     Transformer       │
          │             │          │         110M           │
          │             └──────────┼──────────┘             │
          │                        ▼                        │
          └─────────────── Grounding Checker ───────────────┘
                                   │
                                 Answer
                                   │
                             Apple Neural TTS
                                   │
                             VR Headset Audio
```

---

## Supported Models & Roles

### 1. OpenBioLLM-8B (`richardyoung/openbiollm:latest`)
- **Provider:** Ollama (HTTP REST on `http://127.0.0.1:11434`)
- **Parameters:** ~8 Billion
- **Role:** **Default Production Candidate**
- **Capabilities:** State-of-the-art medical knowledge, highly fluent clinical explanations, phlebotomy protocol compliance.

### 2. Llama 3.2 3B (`llama3.2:3b`)
- **Provider:** Ollama (HTTP REST on `http://127.0.0.1:11434`)
- **Parameters:** ~3.21 Billion
- **Role:** **Benchmark & Fallback Candidate**
- **Capabilities:** Ultra-fast latency (~200ms), low memory footprint, safe clinical refusal under low evidence.

### 3. MedicalTransformerLM 110M (`MedicalTransformerLM`)
- **Provider:** PyTorch (`checkpoints/best_v3.pt`)
- **Parameters:** 110,041,216 (110.04M)
- **Role:** **Preserved Research / Offline Asset**
- **Capabilities:** Runs 100% locally on Apple Silicon MPS via PyTorch. Serves for offline research, model comparison, ablation studies, and benchmarking.
- **Protection Status:** Protected research asset. Checkpoints (`best.pt`, `best_v2.pt`, `best_v3.pt`) are hashed and verified via `outputs/checkpoint_inventory.json` and `scripts/test_medical_transformer_preservation.py`.

---

## Model Selection & AUTO Fallback Matrix

Model selection is configured via `config/models.json` or HTTP request parameters (`POST /ask`).

| Selection Option | Primary Target | Fallback 1 | Fallback 2 | Description |
| :--- | :--- | :--- | :--- | :--- |
| `AUTO` (Default) | OpenBioLLM-8B | Llama 3.2 3B | MedicalTransformerLM 110M | Checks Ollama availability; uses OpenBioLLM if online, falls back safely to Llama 3.2, or PyTorch 110M offline. |
| `OPENBIOLLM` | OpenBioLLM-8B | None | None | Explicitly invokes OpenBioLLM-8B. |
| `LLAMA32` | Llama 3.2 3B | None | None | Explicitly invokes Llama 3.2 3B. |
| `MEDICAL_TRANSFORMER_110M` | MedicalTransformerLM 110M | None | None | Explicitly invokes local PyTorch 110M model. |

> [!IMPORTANT]
> **Model Provenance Transparency:** The backend **never** silently claims one model generated an answer when another model was used. The reported `provider` and `model` fields in response JSON accurately match execution.

---

## VR Simulation Deterministic Boundary

Deterministic VR questions are **never** routed to an LLM.

### Examples of Deterministic VR Queries:
- *"What should I do next?"*
- *"What is step 11?"*
- *"Which object should I pick up?"*
- *"Why was that interaction wrong?"*

### Enforced Isolation Rules:
- Deterministic queries are answered directly by `IntentRouter` + `StepManager` + `VR_GROUND_TRUTH_MAPPING`.
- **The LLM is NEVER allowed to:**
  1. Call `StepManager.Next()`
  2. Modify `CurrentStep`
  3. Mark a step correct or incorrect
  4. Skip a step in the VR workflow

---

## Offline Support & Definitions

- **Fully Offline Local Execution:**
  - Both **OpenBioLLM-8B** and **Llama 3.2 3B** run locally through the local Ollama daemon (`127.0.0.1:11434`).
  - **MedicalTransformerLM 110M** runs locally through PyTorch MPS/CPU execution.
  - Zero external cloud LLM APIs (OpenAI, Anthropic, Google Cloud) are required for any model pathway.
