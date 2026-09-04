#!/usr/bin/env python3
"""
Phase 5 Verification Script: Test Transformer Embeddings
Tests:
1. Initialization & parameter shapes.
2. Forward pass output dimensions (B, T, D).
3. Execution on MPS device (Apple Silicon GPU).
4. Gradient backward pass verification.
5. Context length assertion check.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.embeddings import TransformerEmbeddings
from configs.model_config import ModelConfig

def main():
    print("=" * 60)
    print("PHASE 5: Transformer Embeddings Verification")
    print("=" * 60)
    
    cfg = ModelConfig(vocab_size=16000, embedding_dim=384, context_length=512, dropout=0.1)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Instantiate Module
    emb_layer = TransformerEmbeddings(
        vocab_size=cfg.vocab_size,
        embedding_dim=cfg.embedding_dim,
        context_length=cfg.context_length,
        dropout=cfg.dropout
    ).to(device)
    
    print(f"Module Instantiated: {emb_layer}")
    
    # Check parameters
    tok_params = cfg.vocab_size * cfg.embedding_dim
    pos_params = cfg.context_length * cfg.embedding_dim
    total_emb_params = sum(p.numel() for p in emb_layer.parameters())
    print(f"Token Embedding Params:    {tok_params:,}")
    print(f"Positional Embedding Params:{pos_params:,}")
    print(f"Total Embedding Params:     {total_emb_params:,}")
    assert total_emb_params == tok_params + pos_params, "Parameter count mismatch!"
    
    # 2. Forward Pass Test
    batch_size = 4
    seq_len = 128
    
    # Random token IDs in range [0, V-1]
    dummy_input = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    print(f"\nDummy Input Shape (x): {dummy_input.shape} on {dummy_input.device}")
    
    out = emb_layer(dummy_input)
    print(f"Output Tensor Shape:   {out.shape} on {out.device}")
    
    expected_shape = (batch_size, seq_len, cfg.embedding_dim)
    assert out.shape == expected_shape, f"Expected output shape {expected_shape}, got {out.shape}"
    
    # 3. Backward Pass & Gradient Flow Verification
    print("\n--- Testing Backward Pass & Gradient Flow ---")
    loss = out.sum()
    loss.backward()
    
    tok_grad = emb_layer.token_embeddings.weight.grad
    pos_grad = emb_layer.position_embeddings.weight.grad
    
    assert tok_grad is not None, "Token embedding gradient is None!"
    assert pos_grad is not None, "Position embedding gradient is None!"
    
    print("Token Embedding Gradients:    PRESENT")
    print("Positional Embedding Gradients: PRESENT")
    
    # 4. Assert Context Length Overflow
    print("\n--- Testing Context Length Exceeded Exception ---")
    overflow_input = torch.randint(0, cfg.vocab_size, (1, cfg.context_length + 10), device=device)
    try:
        _ = emb_layer(overflow_input)
        print("[!] ERROR: Overflow assertion failed!")
        return False
    except AssertionError as e:
        print(f"Caught Expected Overflow Exception: {e}")
        
    print("\n" + "=" * 60)
    print("Phase 5 verification PASSED successfully.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
