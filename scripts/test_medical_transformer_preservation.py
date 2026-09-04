#!/usr/bin/env python3
"""
Regression Test Script: 110M MedicalTransformerLM Preservation
Verifies that the existing MedicalTransformerLM architecture, tokenizer,
and checkpoints (best.pt, best_v2.pt, best_v3.pt) remain completely intact,
unmodified, and runnable.
"""

import os
import sys
import json
import hashlib
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from tokenizer.tokenizer import MedicalTokenizer
from configs.model_config import ModelConfig
from training.checkpoint import load_checkpoint
from inference.generate import MedicalGenerator


def calculate_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_preservation():
    print("=" * 70)
    print("110M MEDICALTRANSFORMERLM PRESERVATION & REGRESSION TEST")
    print("=" * 70)

    # 1. Verify Checkpoints & Hashes
    inventory_path = "outputs/checkpoint_inventory.json"
    if not os.path.exists(inventory_path):
        print(f"[-] FAIL: Checkpoint inventory not found at {inventory_path}")
        sys.exit(1)

    with open(inventory_path, "r") as f:
        inventory = json.load(f)

    checkpoints_to_test = ["best.pt", "best_v2.pt", "best_v3.pt"]
    for ckpt_name in checkpoints_to_test:
        ckpt_path = os.path.join("checkpoints", ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[-] FAIL: Checkpoint {ckpt_path} missing!")
            sys.exit(1)

        current_hash = calculate_sha256(ckpt_path)
        expected_hash = inventory.get(ckpt_name, {}).get("sha256")
        
        if current_hash != expected_hash:
            print(f"[-] FAIL: Checkpoint {ckpt_name} SHA-256 mismatch!")
            print(f"    Expected: {expected_hash}")
            print(f"    Actual:   {current_hash}")
            sys.exit(1)
            
        print(f"[✓] Checkpoint {ckpt_name} verified (SHA-256 match).")

    # 2. Verify Tokenizer Loading
    tokenizer_path = "tokenizer/artifacts/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"[-] FAIL: Tokenizer file {tokenizer_path} missing!")
        sys.exit(1)

    tokenizer = MedicalTokenizer(tokenizer_path)
    assert tokenizer.vocab_size == 16000, f"Expected vocab_size 16000, got {tokenizer.vocab_size}"
    test_tokens = tokenizer.encode("What is venipuncture?")
    decoded = tokenizer.decode(test_tokens)
    print(f"[✓] Tokenizer loaded successfully (Vocab Size: {tokenizer.vocab_size}). Test encode/decode working.")

    # 3. Verify Model Architecture Hyperparameters
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        context_length=512
    )
    model = MedicalTransformerLM(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] MedicalTransformerLM initialized. Parameter count: {num_params:,}")
    assert 109_000_000 <= num_params <= 111_000_000, f"Unexpected parameter count: {num_params}"

    # 4. Load Checkpoint (best_v3.pt) and Test Inference
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Loading best_v3.pt checkpoint onto device: {device}...")
    load_checkpoint("checkpoints/best_v3.pt", model, device=device)
    generator = MedicalGenerator(model, tokenizer, device)

    prompt = "Instruction: What is venipuncture?\nMedical Answer:"
    response = generator.generate(
        prompt=prompt,
        max_new_tokens=40,
        temperature=0.7,
        top_k=40,
        top_p=0.9
    )
    print(f"[✓] Inference successful. Generated response snippet:\n    {response[:100]}...")

    print("=" * 70)
    print("[SUCCESS] ALL 110M MEDICALTRANSFORMERLM PRESERVATION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_preservation()
