#!/usr/bin/env python3
"""
Training Configuration Dataclass for Medical LLM.
"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    warmup_steps: int = 50
    max_steps: int = 1000
    eval_interval: int = 100
    checkpoint_interval: int = 200
    seed: int = 42
    device: str = "mps"
    resume_path: str = None
    debug: bool = False
