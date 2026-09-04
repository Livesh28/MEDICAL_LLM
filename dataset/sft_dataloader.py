#!/usr/bin/env python3
"""
Phase 6 Module: Supervised Fine-Tuning (SFT) PyTorch DataLoader
Implements Instruction Loss Masking (-100 for prompt tokens) and 80/10/10 reproducible train/val/test splits.
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import MedicalTokenizer

class VenipunctureSFTDataset(Dataset):
    """
    Supervised Fine-Tuning Dataset for Venipuncture Instructions.
    Applies -100 loss masking to prompt tokens so loss is computed exclusively on response tokens.
    """
    def __init__(self, data_records: List[Dict[str, Any]], tokenizer: MedicalTokenizer, context_length: int = 512):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.samples = []
        
        for item in data_records:
            instruction = item["instruction"]
            inp = item.get("input", "")
            output = item["output"]
            
            if inp:
                prompt_text = f"Instruction: {instruction}\nContext/Input: {inp}\nMedical Answer: "
            else:
                prompt_text = f"Instruction: {instruction}\nMedical Answer: "
                
            response_text = f"{output}"
            
            prompt_ids = tokenizer.encode(prompt_text)
            response_ids = tokenizer.encode(response_text)
            if tokenizer.eot_id is not None:
                response_ids.append(tokenizer.eot_id)
                
            combined_ids = prompt_ids + response_ids
            
            if len(combined_ids) > context_length:
                combined_ids = combined_ids[:context_length]
                
            # Create target labels with -100 for prompt tokens
            prompt_len = min(len(prompt_ids), len(combined_ids))
            labels = [-100] * prompt_len + combined_ids[prompt_len:]
            
            # Pad to context_length
            pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0
            pad_len = context_length - len(combined_ids)
            
            input_ids = combined_ids + [pad_id] * pad_len
            label_ids = labels + [-100] * pad_len
            
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["input_ids"], self.samples[idx]["labels"]

def get_sft_dataloaders(
    json_path: str = "data/dataset_v2/venipuncture_sft_dataset.json",
    tokenizer_path: str = "tokenizer/artifacts/tokenizer.json",
    context_length: int = 512,
    batch_size: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads SFT dataset, performs reproducible 80/10/10 split, and returns PyTorch DataLoaders.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"SFT Dataset not found at {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    tokenizer = MedicalTokenizer(tokenizer_path)
    
    np.random.seed(seed)
    indices = np.random.permutation(len(records))
    
    n_total = len(records)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    
    train_records = [records[i] for i in indices[:n_train]]
    val_records = [records[i] for i in indices[n_train:n_train + n_val]]
    test_records = [records[i] for i in indices[n_train + n_val:]]
    
    train_ds = VenipunctureSFTDataset(train_records, tokenizer, context_length=context_length)
    val_ds = VenipunctureSFTDataset(val_records, tokenizer, context_length=context_length)
    test_ds = VenipunctureSFTDataset(test_records, tokenizer, context_length=context_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print(f"[+] SFT Dataset Split ({n_total} records): {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test.")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    t_loader, v_loader, test_loader = get_sft_dataloaders()
    x, y = next(iter(t_loader))
    print(f"Sample Batch Shapes -> input_ids: {x.shape}, labels: {y.shape}")
    print(f"Unmasked Target Token Count: {(y != -100).sum().item()}")
