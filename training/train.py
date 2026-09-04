#!/usr/bin/env python3
"""
Phase 9 Module: Complete Autoregressive Training Loop & Debug Overfitting System
Includes AdamW optimizer, cosine decay schedule, gradient clipping, MPS acceleration,
loss CSV logging, checkpointing, and a strict --debug single-batch overfitting test.
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
from dataset.dataloader import get_dataloaders
from training.checkpoint import save_checkpoint, load_checkpoint
from training.evaluate import evaluate_model

def get_cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    """
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def run_debug_overfit_test(device: torch.device):
    """
    Overfitting Validation Test on a single mini-batch.
    Ensures loss drops significantly before scaling training.
    """
    print("=" * 60)
    print("RUNNING DEBUG MODE: Single Mini-Batch Overfitting Test")
    print("=" * 60)
    
    cfg = ModelConfig(vocab_size=16000, embedding_dim=384, num_layers=6, context_length=128)
    model = MedicalTransformerLM(cfg).to(device)
    model.train()
    
    model.print_param_summary()
    
    # Create fixed single batch of data
    batch_size = 4
    seq_len = 128
    torch.manual_seed(42)
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    print("\n--- Overfitting Loop (100 Steps) ---")
    initial_loss = 0.0
    final_loss = 0.0
    
    for step in range(1, 101):
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_val = loss.item()
        if step == 1:
            initial_loss = loss_val
        if step == 100:
            final_loss = loss_val
            
        if step == 1 or step % 20 == 0:
            print(f"Step {step:3d}/100 | Overfit Loss: {loss_val:.4f} | Perplexity: {math.exp(loss_val):.2f}")
            
    print(f"\nInitial Step 1 Loss:  {initial_loss:.4f}")
    print(f"Final Step 100 Loss:  {final_loss:.4f}")
    
    # Verify significant loss reduction (>70% drop)
    assert final_loss < initial_loss * 0.3, (
        f"Overfitting test failed! Loss did not decrease significantly. (Initial {initial_loss:.4f} -> Final {final_loss:.4f})"
    )
    
    print(f"OVERFITTING TEST PASSED: Loss reduced from {initial_loss:.4f} down to {final_loss:.4f}")
    
    # Test Checkpoint Save and Resume
    print("\n--- Testing Checkpoint Creation in Debug Mode ---")
    ckpt_path = "checkpoints/debug_test.pt"
    save_checkpoint(ckpt_path, model, optimizer, step=100, train_loss=final_loss, val_loss=final_loss)
    
    resumed_model = MedicalTransformerLM(cfg).to(device)
    load_checkpoint(ckpt_path, resumed_model, device=device)
    
    resumed_logits, resumed_loss = resumed_model(x, targets=y)
    print(f"Resumed Model Verification Loss: {resumed_loss.item():.4f}")
    assert abs(resumed_loss.item() - final_loss) < 1e-4, "Resumed model output mismatch!"
    print("DEBUG OVERFITTING & CHECKPOINT TEST PASSED 100% SUCCESS!")
    print("=" * 60)
    return True

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() and args.device == "mps" else "cpu")
    print(f"Target Execution Device: {device}")
    
    if args.debug:
        run_debug_overfit_test(device)
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. DataLoaders
    print("\n[+] Constructing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        train_bin_path=args.train_bin,
        val_bin_path=args.val_bin,
        context_length=args.context_length,
        batch_size=args.batch_size
    )
    
    # 2. Model Initialization
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=args.context_length,
        dropout=args.dropout
    )
    model = MedicalTransformerLM(cfg).to(device)
    model.print_param_summary()
    
    # 3. Optimizer & Resume
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    start_step = 0
    best_val_loss = float("inf")
    
    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer, device=device)
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        
    # Logging Setup
    log_file = os.path.join(args.output_dir, "training_log.csv")
    write_header = not os.path.exists(log_file) or start_step == 0
    log_csv = open(log_file, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_csv)
    if write_header:
        log_writer.writerow(["step", "train_loss", "val_loss", "perplexity", "learning_rate", "tokens_per_sec", "elapsed_sec"])
        log_csv.flush()
        
    print(f"\n[+] Starting Autoregressive Training Loop from Step {start_step + 1} to {args.max_steps}...")
    model.train()
    
    train_iter = iter(train_loader)
    start_time = time.time()
    accum_loss = 0.0
    
    for step in range(start_step + 1, args.max_steps + 1):
        lr = get_cosine_lr(step, args.max_steps, args.learning_rate, args.min_lr, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        optimizer.zero_grad()
        
        # Gradient Accumulation Sub-Steps
        for micro_step in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
                
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item() * args.grad_accum
            
        # Clip gradients & step optimizer
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        train_loss = accum_loss / args.grad_accum
        accum_loss = 0.0
        
        # Logging & Evaluation
        if step % args.log_interval == 0 or step == args.max_steps:
            eval_res = evaluate_model(model, val_loader, device=device, max_eval_batches=20)
            val_loss = eval_res["val_loss"]
            ppl = eval_res["perplexity"]
            elapsed = time.time() - start_time
            tps = (args.batch_size * args.context_length * args.grad_accum * step) / elapsed if elapsed > 0 else 0
            
            print(
                f"Step {step:5d}/{args.max_steps} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | Speed: {tps:.0f} tok/s",
                flush=True
            )
            
            log_writer.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{ppl:.2f}", f"{lr:.2e}", f"{tps:.0f}", f"{elapsed:.1f}"])
            log_csv.flush()
            
            # Save latest checkpoint
            latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
            save_checkpoint(latest_path, model, optimizer, step=step, train_loss=train_loss, val_loss=val_loss)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, model, optimizer, step=step, train_loss=train_loss, val_loss=val_loss)
                
    log_csv.close()
    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY.")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical LLM")
    parser.add_argument("--debug", action="store_true", help="Run tiny single-batch overfitting debug mode")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--train_bin", type=str, default="data/processed/train_tokens.bin")
    parser.add_argument("--val_bin", type=str, default="data/processed/val_tokens.bin")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    train(args)
