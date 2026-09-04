#!/usr/bin/env python3
"""
Phase 6 Verification Script: Test Causal Multi-Head Self-Attention
Tests:
1. Module initialization & shape validation (B, T, D) -> (B, T, D).
2. MPS hardware execution.
3. Strict Causal Masking Verification:
   Ensures token at position i NEVER attends to any token at position j > i.
4. Gradient flow check for QKV and Output projections.
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.attention import CausalSelfAttention
from configs.model_config import ModelConfig

def main():
    print("=" * 60)
    print("PHASE 6: Causal Multi-Head Self-Attention Verification")
    print("=" * 60)
    
    cfg = ModelConfig(embedding_dim=384, num_heads=6, context_length=512, dropout=0.1)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Selected Device: {device}")
    
    attn = CausalSelfAttention(
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        context_length=cfg.context_length,
        dropout=cfg.dropout
    ).to(device)
    
    print(f"Module Instantiated: {attn}")
    
    total_attn_params = sum(p.numel() for p in attn.parameters())
    print(f"Total Attention Layer Parameters: {total_attn_params:,}")
    
    # 1. Forward Pass Test
    batch_size = 4
    seq_len = 16  # Test sequence length
    
    x = torch.randn(batch_size, seq_len, cfg.embedding_dim, device=device)
    print(f"\nInput Tensor Shape (x):  {x.shape} on {x.device}")
    
    out = attn(x)
    print(f"Output Tensor Shape:    {out.shape} on {out.device}")
    
    assert out.shape == (batch_size, seq_len, cfg.embedding_dim), "Output shape mismatch!"
    
    # 2. Strict Causal Masking Test
    print("\n--- Verifying Causal Masking (Future Tokens Blocked) ---")
    # Manually compute attention weights without dropout to check mask values
    qkv = attn.c_attn(x)
    q, k, v = qkv.split(cfg.embedding_dim, dim=2)
    
    B, T, D = x.size()
    head_dim = cfg.embedding_dim // cfg.num_heads
    
    q = q.view(B, T, cfg.num_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, cfg.num_heads, head_dim).transpose(1, 2)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = attn.causal_mask[:, :, :T, :T]
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.softmax(masked_scores, dim=-1)
    
    # Check that for every head and every batch, attn_weights[b, h, i, j] == 0 for j > i
    for b in range(B):
        for h in range(cfg.num_heads):
            for i in range(T):
                for j in range(i + 1, T):
                    val = attn_weights[b, h, i, j].item()
                    assert val == 0.0, f"Causal mask failure at batch {b}, head {h}, pos {i} -> future pos {j} (val={val})"
                    
    print(f"Causal Masking Check: PASSED (Position i cannot attend to position j > i for all {T} positions)")
    
    # 3. Gradient Flow Test
    print("\n--- Testing Backward Pass & Gradient Flow ---")
    loss = out.sum()
    loss.backward()
    
    c_attn_grad = attn.c_attn.weight.grad
    c_proj_grad = attn.c_proj.weight.grad
    
    assert c_attn_grad is not None, "QKV projection gradient missing!"
    assert c_proj_grad is not None, "Output projection gradient missing!"
    
    print("QKV Projection Gradients:    PRESENT")
    print("Output Projection Gradients: PRESENT")
    
    print("\n" + "=" * 60)
    print("Phase 6 verification PASSED successfully.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
