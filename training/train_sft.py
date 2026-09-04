#!/usr/bin/env python3
"""
Phase 5 & 7 Module: Supervised Fine-Tuning (SFT) Training Pipeline
Continues fine-tuning MedicalTransformerLM (110.04M) on venipuncture SFT instructions with instruction loss masking.
Outputs improved checkpoint to checkpoints/best_v2.pt while preserving original checkpoints/best.pt.
"""

import os
import sys
import time
import math
import csv
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from configs.model_config import ModelConfig
from tokenizer.tokenizer import MedicalTokenizer
from dataset.sft_dataloader import get_sft_dataloaders
from training.checkpoint import save_checkpoint, load_checkpoint

def get_cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def evaluate_sft(model: nn.Module, val_loader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            n_tokens = (shift_labels != -100).sum().item()
            
            if n_tokens > 0:
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
                
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return {"val_loss": avg_loss, "perplexity": ppl}

def train_sft(args):
    device = torch.device("mps" if torch.backends.mps.is_available() and args.device == "mps" else "cpu")
    print("=" * 70)
    print("PHASE 5 & 7: SUPERVISED INSTRUCTION FINE-TUNING (SFT)")
    print(f"Target Execution Device: {device}")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Load DataLoaders
    train_loader, val_loader, test_loader = get_sft_dataloaders(
        json_path=args.sft_data,
        tokenizer_path=args.tokenizer_path,
        context_length=args.context_length,
        batch_size=args.batch_size
    )
    
    # 2. Model Setup
    tokenizer = MedicalTokenizer(args.tokenizer_path)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        context_length=args.context_length,
        dropout=0.1
    )
    model = MedicalTransformerLM(cfg).to(device)
    
    # 3. Load Pre-trained Base Checkpoint
    if os.path.exists(args.base_checkpoint):
        print(f"\n[+] Loading pre-trained base model from: {args.base_checkpoint}")
        load_checkpoint(args.base_checkpoint, model, device=device)
    else:
        print(f"[!] Base checkpoint {args.base_checkpoint} not found. Starting from random initialization.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    log_file = os.path.join(args.output_dir, "sft_training_log.csv")
    log_csv = open(log_file, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_csv)
    log_writer.writerow(["epoch", "step", "train_loss", "val_loss", "perplexity", "learning_rate"])
    
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_v2.pt")
    
    total_steps = len(train_loader) * args.epochs
    step = 0
    start_time = time.time()
    
    print(f"\n[+] Starting SFT Training Loop for {args.epochs} Epochs ({total_steps} Steps)...")
    model.train()
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        
        for x, y in train_loader:
            step += 1
            lr = get_cosine_lr(step, total_steps, args.learning_rate, args.min_lr, warmup_steps=10)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(x)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            n_tokens = (shift_labels != -100).sum().item()
            epoch_loss += loss.item() * max(n_tokens, 1)
            epoch_tokens += max(n_tokens, 1)
            
        avg_train_loss = epoch_loss / max(epoch_tokens, 1)
        eval_res = evaluate_sft(model, val_loader, device=device)
        val_loss = eval_res["val_loss"]
        ppl = eval_res["perplexity"]
        
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | Step {step:3d}/{total_steps} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}"
        )
        
        log_writer.writerow([epoch, step, f"{avg_train_loss:.4f}", f"{val_loss:.4f}", f"{ppl:.2f}", f"{lr:.2e}"])
        log_csv.flush()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt_path, model, optimizer, step=step, train_loss=avg_train_loss, val_loss=val_loss)
            print(f"  [★] Saved new best SFT checkpoint to: {best_ckpt_path} (Val Loss: {val_loss:.4f})")
            
    log_csv.close()
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"SFT TRAINING FINISHED in {elapsed:.1f} seconds.")
    print(f"Best SFT Checkpoint: {best_ckpt_path} (Best Val Loss: {best_val_loss:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT Venipuncture Model")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--sft_data", type=str, default="data/dataset_v2/venipuncture_sft_dataset.json")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    train_sft(args)
