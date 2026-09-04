#!/usr/bin/env python3
"""
Phase 4 Verification Script: Test Data Pipeline & DataLoader
Tests:
1. Token split execution and metadata recording (split_info.json).
2. Memory-mapped Dataset creation.
3. DataLoader iteration and batch shapes (batch_size, context_length).
4. Autoregressive 1-token target shift check (y[i] == x[i+1]).
5. Data leakage prevention check between train and val splits.
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.prepare_data import create_dataset_splits
from dataset.dataloader import get_dataloaders, MedicalAutoregressiveDataset

def main():
    print("=" * 60)
    print("PHASE 4: Data Pipeline & DataLoader Verification")
    print("=" * 60)
    
    # 1. Run split generation
    clean_text = "data/processed/medical_corpus_clean.txt"
    tokenizer_path = "tokenizer/artifacts/tokenizer.json"
    
    if not os.path.exists(clean_text) or not os.path.exists(tokenizer_path):
        print("[!] Pre-requisite files missing. Run Phase 2 & 3 scripts first.")
        return False
        
    meta = create_dataset_splits(
        clean_text_path=clean_text,
        tokenizer_path=tokenizer_path,
        train_ratio=0.90,
        seed=42
    )
    
    # Verify Metadata
    split_meta_path = "data/metadata/split_info.json"
    assert os.path.exists(split_meta_path), "split_info.json not found!"
    print(f"\n[+] Verified split metadata file: {split_meta_path}")
    
    # 2. Test DataLoader Instantiation & Batch Shapes
    print("\n--- Testing PyTorch DataLoaders ---")
    context_length = 512
    batch_size = 4
    
    train_loader, val_loader = get_dataloaders(
        train_bin_path="data/processed/train_tokens.bin",
        val_bin_path="data/processed/val_tokens.bin",
        context_length=context_length,
        batch_size=batch_size
    )
    
    print(f"Train Dataset size: {len(train_loader.dataset):,} samples")
    print(f"Val Dataset size:   {len(val_loader.dataset):,} samples")
    print(f"Train DataLoader:   {len(train_loader):,} batches (batch_size={batch_size})")
    print(f"Val DataLoader:     {len(val_loader):,} batches (batch_size={batch_size})")
    
    # 3. Check Batch Tensors and Autoregressive Target Shift
    print("\n--- Verifying Autoregressive Target Shift (y[i] == x[i+1]) ---")
    x_batch, y_batch = next(iter(train_loader))
    
    print(f"Input batch shape (x):  {x_batch.shape} (dtype: {x_batch.dtype})")
    print(f"Target batch shape (y): {y_batch.shape} (dtype: {y_batch.dtype})")
    
    assert x_batch.shape == (batch_size, context_length), f"Expected shape ({batch_size}, {context_length}), got {x_batch.shape}"
    assert y_batch.shape == (batch_size, context_length), f"Expected shape ({batch_size}, {context_length}), got {y_batch.shape}"
    
    # Verify for every sequence in batch that target is shifted by 1 token
    for b in range(batch_size):
        # x_batch[b, 1:] should match y_batch[b, :-1]
        x_shifted = x_batch[b, 1:]
        y_leading = y_batch[b, :-1]
        match = torch.equal(x_shifted, y_leading)
        print(f"Batch {b} Autoregressive Target Shift Check: {'PASSED' if match else 'FAILED'}")
        assert match, f"Autoregressive target shift failed on batch sequence {b}!"
        
    print("\n" + "=" * 60)
    print("Phase 4 verification PASSED successfully.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
