#!/usr/bin/env python3
"""
Phase 4 Script: Tokenize Corpus & Create Document-Level Train/Val Splits
Performs document-level 90/10 train/validation split to avoid data leakage.
Tokenizes clean text using MedicalTokenizer and saves memory-mapped uint16 binary arrays.
Outputs data/processed/train_tokens.bin and data/processed/val_tokens.bin along with metadata.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import MedicalTokenizer

def create_dataset_splits(
    clean_text_path: str = "data/processed/medical_corpus_clean.txt",
    tokenizer_path: str = "tokenizer/artifacts/tokenizer.json",
    processed_dir: str = "data/processed",
    metadata_dir: str = "data/metadata",
    train_ratio: float = 0.90,
    seed: int = 42
):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    if not os.path.exists(clean_text_path):
        raise FileNotFoundError(f"Clean corpus file not found at {clean_text_path}. Complete Phase 2 first.")
        
    print("=" * 60)
    print("PHASE 4: Document Tokenization & Train/Val Dataset Split")
    print("=" * 60)
    
    tokenizer = MedicalTokenizer(tokenizer_path)
    print(f"Loaded tokenizer (vocab size: {tokenizer.vocab_size:,})")
    
    # Read documents separated by <|endoftext|>
    with open(clean_text_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    # Clean template headers from full text
    import re
    full_text = re.sub(r'Instruction: Answer this question truthfully\n?', '', full_text)
    full_text = re.sub(r'Instruction: [^\n]+\n?', '', full_text)
    full_text = re.sub(r'Context/Input:\s*', '', full_text)
    full_text = re.sub(r'Medical Details:\s*', '', full_text)
    
    documents = [d.strip() for d in full_text.split("<|endoftext|>") if d.strip()]
    print(f"Total documents loaded: {len(documents):,}")
    
    # Document-level shuffle with seed for reproducibility
    np.random.seed(seed)
    indices = np.random.permutation(len(documents))
    
    split_idx = int(len(documents) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_docs = [documents[i] for i in train_indices]
    val_docs = [documents[i] for i in val_indices]
    
    print(f"Document Split: {len(train_docs):,} train docs ({train_ratio*100:.0f}%), {len(val_docs):,} val docs ({(1-train_ratio)*100:.0f}%)")
    
    # Tokenize train documents
    print("\n[+] Tokenizing training documents...")
    train_tokens = []
    for doc in train_docs:
        ids = tokenizer.encode(doc)
        if tokenizer.eot_id is not None:
            ids.append(tokenizer.eot_id)
        train_tokens.extend(ids)
        
    # Tokenize val documents
    print("[+] Tokenizing validation documents...")
    val_tokens = []
    for doc in val_docs:
        ids = tokenizer.encode(doc)
        if tokenizer.eot_id is not None:
            ids.append(tokenizer.eot_id)
        val_tokens.extend(ids)
        
    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)
    
    # Save binary memory-mapped arrays
    train_bin_path = os.path.join(processed_dir, "train_tokens.bin")
    val_bin_path = os.path.join(processed_dir, "val_tokens.bin")
    
    train_arr.tofile(train_bin_path)
    val_arr.tofile(val_bin_path)
    
    # Record metadata
    metadata = {
        "seed": seed,
        "vocab_size": tokenizer.vocab_size,
        "total_documents": len(documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_tokens": len(train_tokens) + len(val_tokens),
        "train_ratio": train_ratio,
        "dtype": "uint16",
        "train_bin_path": train_bin_path,
        "val_bin_path": val_bin_path
    }
    
    split_meta_file = os.path.join(metadata_dir, "split_info.json")
    with open(split_meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("\n--- Dataset Split Summary ---")
    print(f"Train Tokens:      {len(train_tokens):,} tokens ({os.path.getsize(train_bin_path)/1e6:.2f} MB)")
    print(f"Val Tokens:        {len(val_tokens):,} tokens ({os.path.getsize(val_bin_path)/1e6:.2f} MB)")
    print(f"Total Tokens:      {len(train_tokens) + len(val_tokens):,} tokens")
    print(f"Metadata Saved To: {split_meta_file}")
    print("=" * 60)
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Document Token Split Arrays")
    parser.add_argument("--clean_text_path", type=str, default="data/processed/medical_corpus_clean.txt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata")
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    create_dataset_splits(
        clean_text_path=args.clean_text_path,
        tokenizer_path=args.tokenizer_path,
        processed_dir=args.processed_dir,
        metadata_dir=args.metadata_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
