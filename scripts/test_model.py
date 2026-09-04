#!/usr/bin/env python3
"""
Phase 8 Verification Script: Test Full ~30M Parameter Medical Transformer LM
Tests:
1. Model instantiation & parameter count summary (~30.25M parameters).
2. Forward pass with token sequence input (B, T) -> Logits (B, T, V).
3. Cross-entropy loss calculation with targets.
4. Backward pass on MPS device (Apple Silicon GPU).
5. Output shape & logit range checks.
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from configs.model_config import ModelConfig

def main():
    print("=" * 60)
    print("PHASE 8: Full Medical Transformer LM (~30M Params) Verification")
    print("=" * 60)
    
    # Target Architecture Configuration
    cfg = ModelConfig(
        vocab_size=16000,
        embedding_dim=384,
        num_layers=10,
        num_heads=6,
        context_length=512,
        dropout=0.1,
        ff_multiplier=4
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Selected Hardware Acceleration Device: {device}")
    
    # 1. Instantiate Full Model
    print("\n[+] Constructing MedicalTransformerLM Model...")
    model = MedicalTransformerLM(cfg).to(device)
    
    # 2. Print Parameter Breakdown
    model.print_param_summary()
    
    total_params = model.get_num_params()
    # Check that model parameter count is close to 30M
    assert 28_000_000 <= total_params <= 32_000_000, f"Parameter count {total_params} not in expected ~30M range!"
    print(f"Parameter count check PASSED: Model contains exact {total_params:,} parameters (~{total_params/1e6:.2f}M).")
    
    # 3. Forward Pass & Logits Test
    print("\n--- Testing Forward Pass & Logits Shape ---")
    batch_size = 4
    seq_len = 128
    
    # Dummy input token IDs in range [0, V-1]
    dummy_x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    print(f"Input Tokens Tensor (x): {dummy_x.shape} on {dummy_x.device}")
    
    logits, loss = model(dummy_x)
    print(f"Output Logits Tensor:   {logits.shape} on {logits.device}")
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size), f"Expected logits shape {(batch_size, seq_len, cfg.vocab_size)}, got {logits.shape}"
    assert loss is None, "Loss should be None when targets are not provided."
    
    # 4. Forward Pass with Targets & Loss Calculation
    print("\n--- Testing Cross-Entropy Loss & Autoregressive Target Shift ---")
    dummy_y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    
    logits, loss = model(dummy_x, targets=dummy_y)
    print(f"Cross-Entropy Loss:     {loss.item():.4f}")
    assert loss is not None and not torch.isnan(loss), "Loss is invalid or NaN!"
    
    # Initial loss for random weights & uniform distribution over V=16,000 should be ~ log(16000) = 9.68
    expected_init_loss = math.log(cfg.vocab_size)
    print(f"Expected Initial Loss:  ~{expected_init_loss:.2f} (log({cfg.vocab_size}))")
    assert abs(loss.item() - expected_init_loss) < 1.5, f"Initial loss {loss.item()} deviated significantly from {expected_init_loss}"
    
    # 5. Backward Pass & MPS Execution Verification
    print("\n--- Testing Backward Pass & Gradient Updates on MPS ---")
    loss.backward()
    
    missing_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            missing_grads.append(name)
            
    if missing_grads:
        print(f"[!] Warning: Missing gradients for {len(missing_grads)} parameters: {missing_grads}")
        return False
    else:
        print("Backward Pass Success: All model parameters received valid gradients on MPS!")
        
    print("\n" + "=" * 60)
    print("Phase 8 verification PASSED successfully.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
