#!/usr/bin/env python3
"""
Phase 4 Module: Memory-Efficient PyTorch Dataset & DataLoader
Uses memory-mapped binary files (np.memmap) to efficiently sample sequence inputs (x)
and 1-token shifted autoregressive target pairs (y) without loading entire datasets into RAM.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MedicalAutoregressiveDataset(Dataset):
    """
    PyTorch Dataset for autoregressive language modeling.
    Uses memory mapping (memmap) for low memory footprint on Apple Silicon (16GB RAM).
    
    Given sequence length N:
      Input (x):  tokens[i : i + N]
      Target (y): tokens[i + 1 : i + N + 1]
    """
    def __init__(self, bin_path: str, context_length: int = 512, dtype=np.uint16):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Token binary file not found at {bin_path}. Run prepare_data.py first.")
            
        self.bin_path = bin_path
        self.context_length = context_length
        self.dtype = dtype
        
        # Memory-map the binary file (zero RAM copy overhead)
        file_size_bytes = os.path.getsize(bin_path)
        item_size_bytes = np.dtype(dtype).itemsize
        self.num_tokens = file_size_bytes // item_size_bytes
        
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        
        # Max valid starting index for sequence + 1 target token
        self.num_samples = self.num_tokens - context_length
        if self.num_samples <= 0:
            raise ValueError(
                f"Binary dataset token count ({self.num_tokens}) is less than context_length + 1 ({context_length + 1})."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Fetch chunk of length context_length + 1
        chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
        
        x = torch.from_numpy(chunk[:self.context_length])
        y = torch.from_numpy(chunk[1 : self.context_length + 1])
        
        return x, y

def get_dataloaders(
    train_bin_path: str = "data/processed/train_tokens.bin",
    val_bin_path: str = "data/processed/val_tokens.bin",
    context_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42
):
    """
    Factory function to construct memory-efficient Training and Validation DataLoaders.
    """
    train_dataset = MedicalAutoregressiveDataset(train_bin_path, context_length=context_length)
    val_dataset = MedicalAutoregressiveDataset(val_bin_path, context_length=context_length)
    
    # Generator with seed for deterministic batch shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test instantiation
    tr_loader, val_loader = get_dataloaders(batch_size=4, context_length=128)
    print(f"Successfully constructed DataLoaders:")
    print(f"  Train batches: {len(tr_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")
    x_sample, y_sample = next(iter(tr_loader))
    print(f"  Input Tensor (x) shape:  {x_sample.shape}, dtype: {x_sample.dtype}")
    print(f"  Target Tensor (y) shape: {y_sample.shape}, dtype: {y_sample.dtype}")
