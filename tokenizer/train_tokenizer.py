#!/usr/bin/env python3
"""
Phase 3 Script: Train Custom Medical Byte-Pair Encoding (BPE) Tokenizer
Trains a BPE tokenizer from scratch on the processed medical corpus.
Configurable vocabulary size (default 16,000) and special tokens.
"""

import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<med_qa>", "<|endoftext|>"]

def train_medical_tokenizer(corpus_path: str, output_dir: str, vocab_size: int = 16000):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Complete Phase 2 first.")
        
    print("=" * 60)
    print("PHASE 3: Training Custom Medical BPE Tokenizer")
    print("=" * 60)
    print(f"Corpus Path:   {corpus_path}")
    print(f"Vocab Size:    {vocab_size:,}")
    print(f"Special Tokens: {SPECIAL_TOKENS}")
    
    # Initialize BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Normalizer & Pre-tokenizer
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    # Configure BPE Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True
    )
    
    # Train tokenizer on the corpus
    print("\n[+] Training ByteLevel BPE tokenizer on medical corpus...")
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    # Save artifacts
    save_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(save_path)
    
    print(f"\nSaved trained tokenizer artifact to: {save_path}")
    print(f"Final Trained Vocab Size: {tokenizer.get_vocab_size():,}")
    print("=" * 60)
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom BPE Tokenizer for Medical LLM")
    parser.add_argument("--corpus_path", type=str, default="data/processed/medical_corpus_clean.txt", help="Path to clean corpus")
    parser.add_argument("--output_dir", type=str, default="tokenizer/artifacts", help="Directory to save tokenizer artifacts")
    parser.add_argument("--vocab_size", type=int, default=16000, help="Target vocabulary size")
    args = parser.parse_args()
    
    train_medical_tokenizer(args.corpus_path, args.output_dir, args.vocab_size)
