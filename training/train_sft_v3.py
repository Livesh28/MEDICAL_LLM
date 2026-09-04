#!/usr/bin/env python3
"""
Phase 13 & 14 Module: MedicalTransformerLM v3 SFT Training Pipeline
Continues fine-tuning 110.04M PyTorch model on data/sft/venipuncture_sft_dataset_v3.json.
Outputs improved checkpoint to checkpoints/best_v3.pt without overwriting best.pt or best_v2.pt.
Generates outputs/training_report_v3.json.
"""

import os
import sys
import time
import math
import json
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from configs.model_config import ModelConfig
from tokenizer.tokenizer import MedicalTokenizer
from dataset.sft_dataloader_v3 import get_sft_v3_dataloaders
from training.checkpoint import save_checkpoint, load_checkpoint

def get_cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def evaluate_sft_v3(model: nn.Module, val_loader, device: torch.device) -> dict:
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

def train_sft_v3(args):
    device = torch.device("mps" if torch.backends.mps.is_available() and args.device == "mps" else "cpu")
    print("=" * 75)
    print("PHASE 13 & 14: MEDICALTRANSFORMERLM V3 SUPERVISED FINE-TUNING")
    print(f"Device: {device} | Base Checkpoint: {args.base_checkpoint}")
    print("=" * 75)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_sft_v3_dataloaders(
        json_path=args.sft_data,
        tokenizer_path=args.tokenizer_path,
        context_length=args.context_length,
        batch_size=args.batch_size
    )
    
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
    
    if os.path.exists(args.base_checkpoint):
        print(f"[+] Resuming from base checkpoint: {args.base_checkpoint}")
        load_checkpoint(args.base_checkpoint, model, device=device)
    else:
        print(f"[!] Warning: Base checkpoint {args.base_checkpoint} not found.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_loss = float("inf")
    v3_ckpt_path = os.path.join(args.checkpoint_dir, "best_v3.pt")
    
    total_steps = len(train_loader) * args.epochs
    step = 0
    start_time = time.time()
    
    epoch_logs = []
    
    print(f"\n[+] Executing SFT v3 Loop for {args.epochs} Epochs ({total_steps} Steps)...")
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
        eval_res = evaluate_sft_v3(model, val_loader, device=device)
        val_loss = eval_res["val_loss"]
        ppl = eval_res["perplexity"]
        
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | Step {step:3d}/{total_steps} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f}"
        )
        
        epoch_logs.append({
            "epoch": epoch,
            "step": step,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "perplexity": round(ppl, 2),
            "learning_rate": lr
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(v3_ckpt_path, model, optimizer, step=step, train_loss=avg_train_loss, val_loss=val_loss)
            print(f"  [★] Saved new best SFT v3 checkpoint -> {v3_ckpt_path} (Val Loss: {val_loss:.4f})")
            
    elapsed = time.time() - start_time
    
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": "MedicalTransformerLM_v3",
        "checkpoint_saved": v3_ckpt_path,
        "epochs": args.epochs,
        "total_steps": total_steps,
        "elapsed_seconds": round(elapsed, 1),
        "best_val_loss": round(best_val_loss, 4),
        "epoch_logs": epoch_logs
    }
    
    report_path = os.path.join(args.output_dir, "training_report_v3.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("=" * 75)
    print(f"SFT V3 TRAINING FINISHED in {elapsed:.1f} seconds.")
    print(f"Best SFT v3 Checkpoint: {v3_ckpt_path} (Best Val Loss: {best_val_loss:.4f})")
    print(f"Training Report Saved:  {report_path}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT v3 Medical Model")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/best_v2.pt")
    parser.add_argument("--sft_data", type=str, default="data/sft/venipuncture_sft_dataset_v3.json")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    train_sft_v3(args)
