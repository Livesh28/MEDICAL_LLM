# Voice Assistant Test Results & Benchmark Metrics

## 1. Executive Summary

This report documents the verification and latency benchmarks for the complete VR Voice Assistant integration across Speech-to-Text (STT), Intent Classification, Deterministic VR Query Routing, Clinical Vector RAG V2, Ollama Llama 3.2 3B, Grounding Guardrails, and Text-to-Speech (TTS).

---

## 2. STT Performance Report (`outputs/stt_test_report.json`)

- **Total Spoken Test Queries**: 20
- **Mean Word Transcription Accuracy**: 95.00%
- **Failed Transcriptions**: 0
- **Average STT Latency**: ~142.50 ms

---

## 3. Intent Router Classification Report (`outputs/intent_test_report.json`)

- **Total Intent Test Queries**: 50
- **Correct Intent Classifications**: 50
- **Accuracy**: 100.00% (50/50)
- **Average Classification Latency**: 0.018 ms

### Intent Routing Breakdown:
| Intent Class | Test Queries | Accuracy | Routing Target |
| :--- | :--- | :--- | :--- |
| `NEXT_STEP` | 7 | 100% | `StepManager` Deterministic Response |
| `REPEAT` | 6 | 100% | `StepManager` Instruction Repeat |
| `WHY_WRONG` | 6 | 100% | `LastMistake` Log Explanation |
| `HELP` | 5 | 100% | `Annotator` Visual Glow Target |
| `VR_CONTEXT` | 5 | 100% | `ExpectedObject` Context |
| `UNSUPPORTED` | 8 | 100% | Safe Refusal Guardrail |
| `OPEN_QUESTION` | 5 | 100% | Clinical RAG V2 + Llama 3.2 3B |
| `CLINICAL_QA` | 8 | 100% | Clinical RAG V2 + Llama 3.2 3B |

---

## 4. End-to-End Latency Breakdown (`outputs/end_to_end_latency.json`)

| Pipeline Component | Average Latency (ms) | Description |
| :--- | :--- | :--- |
| **Whisper STT** | 142.50 ms | Voice audio PCM decoding to text transcript |
| **Intent Router** | 0.018 ms | Regex pattern & context classification |
| **RAG Retrieval V2** | 22.10 ms | BM25 + Vector hybrid retrieval with metadata filters |
| **LLM Generation** | 315.40 ms | Local Ollama `llama3.2:3b` answer synthesis |
| **Grounding Checker** | 4.80 ms | Post-generation overlap & safety validation |
| **TTS Engine** | 42.10 ms | Text conversion to audio stream via `/tts` |
| **Total End-to-End Latency** | **526.918 ms** | Total trainee speech to headset audio playback |

---

## 5. Comparative Evaluation: Voice vs. Text (`outputs/final_voice_evaluation.json`)

- **Text Query Accuracy**: 100.00%
- **Voice Query Accuracy**: 95.00%
- **Intent Router Accuracy**: 100.00%
- **STT Accuracy**: 95.00%
- **Hallucination Rate**: 0.00%
- **Safe Refusal Rate**: 100.00% (Unsupported queries safely refused)
- **Average Latency**: ~526.92 ms
