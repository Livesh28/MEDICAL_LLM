# End-to-End Voice Assistant & VR Architecture

## 1. System Overview

The voice assistant provides real-time phlebotomy guidance and clinical question answering inside the Medical VR Venipuncture Training Simulation.

```mermaid
flowchart TD
    subgraph Unity_VR ["Unity VR Headset Subsystem"]
        Mic[VR Microphone / Push-to-Talk]
        UI[VR Voice UI Overlay]
        Audio[Unity AudioSource Headset Player]
        StepMgr[StepManager / State Core]
    end

    subgraph FastAPI_Backend ["FastAPI Server (http://127.0.0.1:8000)"]
        STT[Whisper STT Service /stt]
        Router{Intent Router api/intent_router.py}
        DetEngine[Deterministic VR Step Engine]
        RAG[Metadata-Aware Hybrid RAG V2]
        LLM[Local Ollama Llama 3.2 3B]
        Checker[Grounding Checker Guardrail]
        TTS[Neural TTS Service /tts]
    end

    Mic -->|WAV Audio Bytes| STT
    STT -->|Recognized Transcript| Router
    StepMgr -.->|CurrentStep & LastMistake| Router

    Router -- "NEXT_STEP / REPEAT / WHY_WRONG / HELP / VR_CONTEXT" --> DetEngine
    Router -- "CLINICAL_QA / OPEN_QUESTION" --> RAG
    Router -- "UNSUPPORTED" --> SafetyRefusal[Safe Refusal Guard]

    RAG -->|Top Evidence Chunks| LLM
    LLM -->|Candidate Answer| Checker

    DetEngine -->|Authoritative Answer Payload| TTS
    Checker -->|Verified Answer Payload| TTS
    SafetyRefusal -->|Safe Refusal Payload| TTS

    TTS -->|Audio Bytes| Audio
    TTS -.->|Answer Text & Status| UI
```

---

## 2. Deterministic VR Core Immutability Guardrail

The C# simulation core components (`StepManager`, `Veni`, `StepList`, `Annotator`, `Grabbable`, `Trigger`, `SnapZone`) remain the sole authoritative source of truth for procedure logic and state progression.

- **Rule 1**: The LLM engine **never** decides or updates `StepManager.CurrentStep`.
- **Rule 2**: Deterministic voice queries (`NEXT_STEP`, `REPEAT`, `WHY_WRONG`, `HELP`, `VR_CONTEXT`) bypass LLM generation entirely and read direct state from `venipuncture_16_steps.json`.
- **Rule 3**: In **Test Mode**, the voice assistant is automatically disabled to evaluate unassisted trainee proficiency.
