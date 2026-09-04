#!/usr/bin/env python3
"""
Phase 9 Module: Checkpoint Manager
Handles saving and loading training state (model, optimizer, scheduler, metrics, config).
"""

import os
import torch
from typing import Optional, Dict, Any

def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    step: int = 0,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    config: Optional[Dict[str, Any]] = None
):
    """
    Saves atomic checkpoint file to disk.
    """
    dir_name = os.path.dirname(checkpoint_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config
    }
    
    # Save atomically via tmp file to prevent corrupt checkpoints
    tmp_path = checkpoint_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    print(f"[+] Saved checkpoint to: {checkpoint_path} (step {step}, val_loss: {val_loss:.4f})")

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Loads checkpoint state into model and optimizer.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        
    print(f"[+] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    print(f"Resumed training state at step {checkpoint.get('step', 0)} (val_loss: {checkpoint.get('val_loss', 0.0):.4f})")
    return checkpoint
