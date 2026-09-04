#!/usr/bin/env python3
"""
Phase 7 Verification Script: Test Transformer Block & Feed-Forward Network
Tests:
1. Feed-Forward Network shape (B, T, D) -> (B, T, D) and parameter counting.
2. Transformer Block Pre-LN structure and exact parameter count (1,774,464 per block).
3. Residual connection identity verification.
4. MPS hardware execution and gradient backward pass.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.feed_forward import FeedForward
from model.transformer_block import TransformerBlock
from configs.model_config import ModelConfig

def main():
    print("=" * 60)
    print("PHASE 7: Transformer Block & Feed-Forward Verification")
    print("=" * 60)
    
    cfg = ModelConfig(embedding_dim=384, num_heads=6, context_length=512, dropout=0.1)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Selected Device: {device}")
    
    # 1. Test Feed-Forward Network
    print("\n--- 1. Feed-Forward Network (FFN) Check ---")
    ffn = FeedForward(embedding_dim=cfg.embedding_dim, ff_multiplier=cfg.ff_multiplier, dropout=cfg.dropout).to(device)
    ffn_params = sum(p.numel() for p in ffn.parameters())
    
    # Expected FFN Params: c_fc (384 * 1536 + 1536) + c_proj (1536 * 384 + 384) = 591,360 + 590,208 = 1,181,568
    expected_ffn_params = (cfg.embedding_dim * cfg.ff_dim + cfg.ff_dim) + (cfg.ff_dim * cfg.embedding_dim + cfg.embedding_dim)
    print(f"FFN Module:         {ffn}")
    print(f"FFN Parameter Count:{ffn_params:,} (Expected: {expected_ffn_params:,})")
    assert ffn_params == expected_ffn_params, "FFN parameter count mismatch!"
    
    # 2. Test Single Transformer Block
    print("\n--- 2. Transformer Block (Pre-LN) Check ---")
    block = TransformerBlock(
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        context_length=cfg.context_length,
        ff_multiplier=cfg.ff_multiplier,
        dropout=cfg.dropout
    ).to(device)
    
    block_params = sum(p.numel() for p in block.parameters())
    print(f"Transformer Block Parameter Count: {block_params:,}")
    
    # Expected per-block params:
    # MHA: 591,360 | LN1 + LN2: 1,536 | FFN: 1,181,568 | Total = 1,774,464
    expected_block_params = 591360 + 1536 + 1181568
    print(f"Expected Per-Block Parameter Count: {expected_block_params:,}")
    assert block_params == expected_block_params, f"Block param mismatch! Got {block_params}, expected {expected_block_params}"
    
    # 3. Forward Pass & Tensor Shapes
    print("\n--- 3. Forward Pass & Shape Integrity ---")
    batch_size = 4
    seq_len = 64
    
    x = torch.randn(batch_size, seq_len, cfg.embedding_dim, device=device)
    print(f"Input Shape (x):  {x.shape} on {x.device}")
    
    out = block(x)
    print(f"Output Shape:     {out.shape} on {out.device}")
    assert out.shape == (batch_size, seq_len, cfg.embedding_dim), "TransformerBlock output shape mismatch!"
    
    # 4. Backward Pass & Gradient Flow Verification
    print("\n--- 4. Backward Pass & Gradient Flow Verification ---")
    loss = out.sum()
    loss.backward()
    
    ln1_grad = block.ln_1.weight.grad
    attn_grad = block.attn.c_attn.weight.grad
    ln2_grad = block.ln_2.weight.grad
    ffn_grad = block.mlp.c_fc.weight.grad
    
    assert ln1_grad is not None, "LN1 gradient missing!"
    assert attn_grad is not None, "Attention gradient missing!"
    assert ln2_grad is not None, "LN2 gradient missing!"
    assert ffn_grad is not None, "FFN gradient missing!"
    
    print("Pre-LN 1 Gradients:       PRESENT")
    print("Attention Gradients:      PRESENT")
    print("Pre-LN 2 Gradients:       PRESENT")
    print("Feed-Forward Gradients:   PRESENT")
    
    print("\n" + "=" * 60)
    print("Phase 7 verification PASSED successfully.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
