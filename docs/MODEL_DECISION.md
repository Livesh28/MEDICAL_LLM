# Multi-Model Production Decision & Architecture Specification

## Executive Decision
The production architecture for the **AI-Enhanced VR Venipuncture Training System** is a **Multi-Model Router Architecture**:

1. **Primary Production Candidate:** **`OpenBioLLM-8B` (`richardyoung/openbiollm:latest` via Ollama) + RAG V2**.
   - **Why:** State-of-the-art medical LLM performance for complex clinical QA, phlebotomy protocol inquiries, and trainee explanations. Combined with RAG V2 hybrid vector retrieval, it delivers grounded, evidence-backed answers.

2. **Benchmark / Fast Fallback:** **`Llama 3.2 3B` (`llama3.2:3b` via Ollama) + RAG V2**.
   - **Why:** Achieves low latency (~200ms) with zero hallucination rate on gold-standard phlebotomy benchmarks under strict grounding constraints.

3. **Preserved Research / Offline Model:** **`MedicalTransformerLM` (PyTorch 110.04M Parameters, [`checkpoints/best_v3.pt`](file:///Users/livesh/Medical_LLM/checkpoints/best_v3.pt))**.
   - **Why:** Preserved as a protected research asset for offline evaluation, model comparison, ablation studies, and benchmarking. Runs 100% locally on PyTorch MPS/CPU without requiring external daemon setup.

4. **VR Simulation State Guardrail:** C# Unity `StepManager` deterministically handles VR step queries (`Grabbable`, `Trigger`, `SnapZone`). The LLM **never** overrides procedural simulation state or advances step numbers.

---

## Production System Flow

```text
Unity VR (StepManager / Grabbable / Trigger / SnapZone)
   ↓
Whisper Speech-to-Text (STT)
   ↓
Intent Router (api/intent_router.py)
   ├── Deterministic VR Question ──> Direct StepManager Response (No LLM)
   └── Clinical QA Question ──────> Model Router (OpenBioLLM-8B / Llama 3.2 / MedicalTransformer 110M)
                                          ↓
                                    RAG V2 Hybrid Retrieval (data/rag_db)
                                          ↓
                                    Selected LLM Engine Generation
                                          ↓
                                    Grounding Validation & Source Metadata
                                          ↓
                                    Apple Neural TTS Voice Assistant
                                          ↓
                                    VR Headset Audio Stream
```
