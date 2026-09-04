# Voice Assistant & Unity VR Integration Guide

## 1. Overview
This document specifies the end-to-end integration of the real-time voice assistant for the Medical VR – Venipuncture Training Simulation codebase.

The voice pipeline connects trainee speech in Unity VR directly to Speech-to-Text (STT), Intent Classification, Deterministic VR State / Clinical RAG, LLM Synthesis, Grounding Guardrails, Text-to-Speech (TTS), and VR Headset AudioSource playback.

---

## 2. Voice Pipeline Architecture

```
VR Microphone (Push-To-Talk)
      ↓
Whisper STT (FastAPI /stt endpoint)
      ↓
Intent Router (api/intent_router.py)
      ├── NEXT_STEP / REPEAT / WHY_WRONG / HELP / VR_CONTEXT → StepManager (Deterministic VR State)
      ├── CLINICAL_QA / OPEN_QUESTION → RAG V2 + Ollama Llama 3.2 3B + Grounding Checker
      └── UNSUPPORTED → Safe Refusal Guardrail
      ↓
Response JSON (answer, engine, grounded, intent, sources, latency breakdown)
      ↓
Neural TTS (FastAPI /tts endpoint)
      ↓
Unity AudioSource Playback in VR Headset
```

---

## 3. Unity C# Scripts

The Unity C# subsystem is organized into 4 modular components:

1. **`VoiceInputManager.cs`** ([`VoiceInputManager.cs`](file:///Users/livesh/Medical_LLM/unity/Scripts/VoiceInputManager.cs)):
   - Handles microphone permissions and device selection.
   - Manages Push-to-Talk audio recording buffer with max duration limits.
   - Detects speech volume thresholds (RMS) and suppresses empty audio.
   - Encodes raw audio PCM samples into WAV byte arrays for transmission.

2. **`VRVoiceAssistant.cs`** ([`VRVoiceAssistant.cs`](file:///Users/livesh/Medical_LLM/unity/Scripts/VRVoiceAssistant.cs)):
   - Orchestrates HTTP POST calls to `/stt` and `/ask`.
   - Passes VR simulation context (`current_step`, `step_name`, `last_mistake`).
   - Guarantees **StepManager Immutability**: AI responses never alter step index or VR state.
   - Automatically **disables Voice Assistant in Test Mode** (`SimulationMode.TestMode`).

3. **`TTSManager.cs`** ([`TTSManager.cs`](file:///Users/livesh/Medical_LLM/unity/Scripts/TTSManager.cs)):
   - Fetches AIFF/WAV audio stream from `/tts`.
   - Dynamically instantiates a Unity `AudioClip`.
   - Plays audio via attached VR headset `AudioSource`.

4. **`VRVoiceUIManager.cs`** ([`VRVoiceUIManager.cs`](file:///Users/livesh/Medical_LLM/unity/Scripts/VRVoiceUIManager.cs)):
   - Renders VR canvas state overlay ("Listening...", "Transcribing...", "Thinking...", Answer text, Playback status).

---

## 4. FastAPI Endpoints & API Contract

### `POST /stt`
- **Payload**: `Multipart/form-data` with `file` (audio `.wav` / `.mp3`).
- **Response**:
  ```json
  {
    "transcript": "What should I do next?",
    "duration_sec": 1.5,
    "stt_ms": 142.5,
    "status": "success",
    "error": null
  }
  ```

### `POST /ask`
- **Request**:
  ```json
  {
    "question": "What should I do next?",
    "current_step": 11,
    "step_name": "Insert Tube",
    "last_mistake": "None",
    "top_k_chunks": 2,
    "temperature": 0.3
  }
  ```
- **Response**:
  ```json
  {
    "question": "What should I do next?",
    "answer": "Insert the Blood Collection Tube into the Tube Slot.",
    "engine": "vr_stepmanager_deterministic",
    "grounded": true,
    "confidence": "high",
    "intent": "NEXT_STEP",
    "sources": [],
    "latency": {
      "stt_ms": 0.0,
      "intent_ms": 0.015,
      "retrieval_ms": 0.0,
      "llm_ms": 0.0,
      "grounding_ms": 0.0,
      "tts_ms": 0.0,
      "total_ms": 0.02
    }
  }
  ```

### `GET /tts`
- **Params**: `text` (str), `voice` (str), `rate` (int WPM).
- **Response**: Binary audio byte stream (`audio/aiff`).
