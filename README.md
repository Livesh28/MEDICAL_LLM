# Local Medical LLM & Collaborative RAG Workbench

> **IMPORTANT CLINICAL SAFETY NOTICE**: This software system is an educational and simulation research prototype built to demonstrate domain-specific medical language modeling, clinical information retrieval, and medical VR training assistants. It is **NOT** a certified medical diagnostic device and **MUST NOT** be used for clinical decision-making, direct emergency patient care, or professional medical diagnosis.

---

## 1. Project Overview

The **Local Medical LLM & RAG Workbench** is an end-to-end, 100% locally executable medical AI workstation designed for clinical question answering, phlebotomy procedure verification (CLSI GP41 & WHO guidelines), and immersive virtual reality medical training.

The system runs entirely on consumer hardware (Apple Silicon MPS / macOS / Linux / CPU) with **zero cloud API dependencies**, ensuring strict medical data privacy and compliance.

### Core Capabilities:
- **Collaborative Multi-Model Pipeline**: Synthesizes verified clinical knowledge by coordinating specialized biomedical extractors with instruction-tuned reasoning engines.
- **CLSI & WHO Clinical Vector RAG**: Hybrid retrieval (BM25 lexical + dense PyTorch vector search with Reciprocal Rank Fusion) over 2,029 verified clinical phlebotomy and procedure chunks.
- **Medical Acronym & Synonym Query Expansion**: Clinical dictionary expansion (`EDTA`, `SST`, `CBC`, `CLSI`, `PT`, `PTT`, etc.) for high-precision laboratory retrieval.
- **Native PyTorch Medical Transformer**: A domain-specific 110.04M parameter decoder-only Transformer trained from scratch on medical corpora.
- **Voice-Enabled Clinical Interface**: Whisper STT audio transcription and browser/Unity Text-to-Speech playback.
- **Interactive Web Studio & Terminal CLI**: Dual interface for both browser chat and terminal execution with real-time token streaming.
- **Native Meta Quest VR Integration**: Complete C# client suite for Unity VR phlebotomy simulation training.

---

## 2. Architecture & Main Components

```
User Query (Text / Audio via Whisper)
                  │
                  ▼
         [ Query Cache (<6ms) ] ──(Hit)──► Instant Response
                  │
                (Miss)
                  │
                  ▼
   [ Medical Acronym Expansion ]
                  │
                  ▼
   [ Hybrid Clinical Retrieval ]
   ├── BM25 Lexical Keyword Search
   └── PyTorch Vector Cosine Similarity
                  │
                  ▼
     [ Reciprocal Rank Fusion ]
                  │
                  ▼
     Top Verified Clinical Chunks
                  │
                  ▼
   [ Multi-Model Pipeline Routing ]
   ├── Unified Collaborative (RAG + OpenBioLLM 8B + Llama 3.2 3B)
   ├── OpenBioLLM-8B (Biomedical Specialist via Ollama)
   ├── Llama 3.2 3B (General Clinical Synthesis via Ollama)
   └── MedicalTransformerLM 110M (Local PyTorch on MPS/CPU)
                  │
                  ▼
    [ Grounding Verification ]
                  │
                  ▼
   [ FastAPI Async Serving ]
   ├── Web UI (Doctor Osler.AI Chat & Settings)
   ├── REST API (/ask, /ask_stream, /stt)
   ├── Interactive Terminal CLI (cli.py)
   └── Unity Meta Quest VR Client
```

---

## 3. Supported Models

| Model | Provider / Engine | Role | Local Checkpoint / Tag |
|---|---|---|---|
| **Unified Collaborative AI** | Multi-Model RAG | Master pipeline: RAG evidence $\rightarrow$ OpenBioLLM extraction $\rightarrow$ Llama 3.2 synthesis | `richardyoung/openbiollm:latest` + `llama3.2:3b` |
| **OpenBioLLM-8B** | Ollama | Biomedical specialist for clinical pharmacology and anatomy | `richardyoung/openbiollm:latest` |
| **Llama 3.2 3B** | Ollama | Fast, fluent clinical structuring and synthesis | `llama3.2:3b` |
| **MedicalTransformerLM 110M** | Local PyTorch (MPS/CPU) | 12-layer Decoder-only Transformer trained from scratch | `checkpoints/best_v3.pt` |

> **Note on Checkpoint Files**: Due to GitHub file size limitations, binary model checkpoints (`*.pt` files in `checkpoints/`) are excluded from version control via `.gitignore`. The local architecture and weights loading logic remain fully preserved in `model/transformer_lm.py` and `training/checkpoint.py`.

---

## 4. Clinical RAG Architecture & Datasets

### Knowledge Sources:
1. **CLSI GP41-Ed7**: Global standard for diagnostic venous blood specimen collection (order of draw, tube additives, needle angle, precautions).
2. **WHO Best Practices in Phlebotomy (2010)**: International infection control and blood draw protocols.
3. **WHO Hand Hygiene Guidelines (2009)**: Strict hygiene and PPE standards.
4. **Medical VR Ground Truth**: Standardized 16-step venipuncture training procedure specification.

### Data Storage & Rebuilding:
- **Source Documents**: Located in `data/clinical_knowledge/`, `data/vr_knowledge/`, `data/rag_sources/`, and `data/metadata/`.
- **Vector Database**: Stored locally in `data/rag_db/` (`chunks.json`, `metadata.pt`, `vectors.pt`).
- **How to Rebuild RAG**: Run the authoritative ingestion pipeline:
  ```bash
  python3 scripts/ingest_authoritative_sources.py
  ```

---

## 5. Repository Directory Structure

```
Medical_LLM/
├── api/                     # FastAPI backend, REST routes, STT service, Intent Router
│   ├── server.py            # Main server & Doctor Osler.AI Web UI
│   ├── intent_router.py     # Deterministic VR vs clinical question classifier
│   └── stt_service.py       # Whisper audio transcription service
├── checkpoints/             # Local PyTorch checkpoints (best_v3.pt, best.pt)
├── cli.py                   # Terminal CLI for real-time interactive chat
├── config/                  # Model provider registry (models.json)
├── configs/                 # Hyperparameters (model_config.py, training_config.py)
├── data/                    # Clinical knowledge, VR specifications, and metadata
│   ├── clinical_knowledge/  # CLSI & WHO structured clinical Q&A
│   ├── vr_knowledge/        # 16-step VR simulation ground truth
│   ├── rag_sources/         # Verified RAG source registry
│   └── metadata/            # Source licenses and dataset provenance
├── dataset/                 # PyTorch dataloaders and memory-mapped dataset pipelines
├── docs/                    # Technical documentation, audits, and architectural specs
├── inference/               # Model providers and autoregressive generation
│   ├── model_provider.py    # Ollama & PyTorch abstraction router with keep-alive
│   └── generate.py          # Autoregressive sampling (Top-K, Top-P, Temperature)
├── model/                   # PyTorch Transformer Architecture
│   ├── transformer_lm.py    # 110.04M MedicalTransformerLM model definition
│   ├── attention.py         # Causal Multi-Head Self-Attention
│   ├── embeddings.py        # Token & Positional Embeddings
│   └── transformer_block.py # Pre-LN Transformer Decoder Block
├── rag/                     # RAG pipeline, hybrid retriever, cache, and chunker
│   ├── pipeline.py          # Master RAG pipeline (Unified, OpenBioLLM, Llama)
│   ├── retriever_v2.py      # Metadata-aware BM25 hybrid retriever & acronym expansion
│   ├── database.py          # PyTorch cosine similarity vector database
│   ├── chunker.py           # Sliding-window document chunker
│   └── cache.py             # Thread-safe in-memory LRU query cache
├── scripts/                 # Unit tests, benchmarks, and data ingestion scripts
├── tokenizer/               # 16,000-vocab ByteLevel BPE tokenizer & artifacts
├── training/                # Training loops, loss computation, and evaluation benchmarks
├── unity/                   # Unity Meta Quest VR C# client scripts
│   └── Scripts/             # VRVoiceAssistant.cs, TTSManager.cs, VoiceInputManager.cs
├── requirements.txt         # Python dependencies
├── .env.example             # Environment configuration template
├── .gitignore               # Git exclusions for checkpoints, vector DBs, and caches
└── LICENSE                  # Open-source license with medical disclaimer
```

---

## 6. Installation & Quick Start

### 1. Prerequisites
- macOS (Apple Silicon M1/M2/M3/M4 recommended) or Linux
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/Medical_LLM.git
cd Medical_LLM

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```

### 4. Pull Ollama Models
Ensure Ollama is running (`ollama serve`), then pull the clinical models:
```bash
ollama pull richardyoung/openbiollm:latest
ollama pull llama3.2:3b
```

---

## 7. How to Run

### Option A: Interactive Web UI (Doctor Osler.AI)
Launch the FastAPI web studio:
```bash
python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```
Open your browser to: **`http://localhost:8000`**
- **Clinical Chat**: The main page is dedicated to clinical inquiry with quick prompts and audio playback.
- **Settings**: Click the **Settings** tab to adjust models, temperature, speech persona, document ingestion, or inspect hardware telemetry.

### Option B: Terminal CLI
Ask questions directly from your terminal with streaming output:
```bash
# Interactive conversation loop
python3 cli.py

# One-shot clinical query
python3 cli.py "What is the CLSI draw order for EDTA and SST tubes?"
```

### Option C: REST API (cURL)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the standard needle insertion angle for venipuncture?",
    "model": "unified",
    "top_k": 4
  }'
```

---

## 8. Verification & Test Suite

Run the automated validation scripts to verify all system components:
```bash
# Verify RAG retrieval and citation grounding
python3 scripts/test_rag.py

# Verify FastAPI endpoints
python3 scripts/test_api.py

# Verify Apple Silicon MPS hardware acceleration
python3 scripts/test_mps.py

# Benchmark retrieval across clinical acronyms
python3 scripts/run_retrieval_matrix.py
```

---

## 9. License & Safety Disclaimer

Distributed under the MIT License. See [`LICENSE`](file:///Users/livesh/Medical_LLM/LICENSE) for complete terms. This software is strictly for simulation, educational, and research purposes.
