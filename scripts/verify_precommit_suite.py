#!/usr/bin/env python3
"""
Pre-Commit Automated System Verification Suite
Validates all checklist items specified in Section 18 of the GitHub Preparation guidelines:
  1. FastAPI server & health endpoint
  2. Model execution: Unified, OpenBioLLM, Llama 3.2, MedicalTransformerLM
  3. Checkpoint verification: best.pt, best_v2.pt, best_v3.pt
  4. Whisper STT initialization
  5. RAG retrieval & verified source output
  6. Frontend HTML integrity
"""

import os
import sys
import requests
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from configs.model_config import ModelConfig
from tokenizer.tokenizer import MedicalTokenizer
from training.checkpoint import load_checkpoint

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("=== 1. Testing Health & Server Status ===")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    data = r.json()
    print(f"✓ Health OK: Status={data.get('status')}, Device={data.get('device')}")

def test_checkpoints():
    print("\n=== 2. Testing Checkpoints (best.pt, best_v2.pt, best_v3.pt) ===")
    tok = MedicalTokenizer("tokenizer/artifacts/tokenizer.json")
    cfg = ModelConfig(vocab_size=tok.vocab_size, embedding_dim=768, num_layers=12, num_heads=12, context_length=512)

    for ckpt_name in ["best.pt", "best_v2.pt", "best_v3.pt"]:
        path = os.path.join("checkpoints", ckpt_name)
        assert os.path.exists(path), f"Checkpoint missing: {path}"
        model = MedicalTransformerLM(cfg)
        state = load_checkpoint(checkpoint_path=path, model=model, device="cpu")
        print(f"✓ Loaded {ckpt_name}: step={state.get('step')}, val_loss={state.get('val_loss')}")

def test_models_and_queries():
    print("\n=== 3. Testing Queries across Models ===")
    test_cases = [
        ("unified", "What is the CLSI draw order for EDTA and SST tubes?"),
        ("openbiollm", "What is Step 0 in the venipuncture training workflow?"),
        ("llama32", "What is the role of StepManager in the VR simulation?"),
        ("medical_transformer_110m", "What is VeinTrigger used for?"),
    ]

    for model, q in test_cases:
        print(f"\nQuerying [{model}] with: '{q}'")
        payload = {
            "question": q,
            "model": model,
            "top_k_chunks": 3,
            "max_new_tokens": 80,
            "temperature": 0.2
        }
        r = requests.post(f"{BASE_URL}/ask", json=payload, timeout=60)
        assert r.status_code == 200, f"Query failed for {model}: {r.status_code} - {r.text}"
        res = r.json()
        ans = res.get("answer", "")
        sources = res.get("sources", [])
        print(f"  ✓ Status 200 OK | Answer length: {len(ans)} chars")
        print(f"  ✓ Answer preview: {ans[:120]}...")
        print(f"  ✓ Sources retrieved: {len(sources)}")

def test_stt():
    print("\n=== 4. Testing Whisper STT Service ===")
    from api.stt_service import WhisperSTTService
    stt = WhisperSTTService(model_name="tiny")
    assert stt.model is not None, "Whisper model not initialized"
    print("✓ Whisper STT Service initialized successfully.")

def test_frontend():
    print("\n=== 5. Testing Frontend HTML Structure ===")
    r = requests.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200
    html = r.text
    assert "Doctor.AI" in html
    assert "panel-chat" in html
    assert "panel-settings" in html
    assert "rag-q" in html
    assert "model-sel" in html
    print("✓ Frontend HTML fully verified.")

if __name__ == "__main__":
    test_health()
    test_checkpoints()
    test_models_and_queries()
    test_stt()
    test_frontend()
    print("\n" + "=" * 50)
    print("ALL PRE-COMMIT VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 50)
