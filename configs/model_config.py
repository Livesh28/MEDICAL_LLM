#!/usr/bin/env python3
"""
Model & Training Configurations for 30M-Parameter Local Medical LLM.
Centralized hyperparameters for architecture, training loop, and system execution.
"""

from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    vocab_size: int = 16000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    context_length: int = 512
    dropout: float = 0.1
    ff_multiplier: int = 4  # ff_dim = 4 * embedding_dim = 3072

    @property
    def ff_dim(self) -> int:
        return self.ff_multiplier * self.embedding_dim

    @property
    def head_dim(self) -> int:
        assert self.embedding_dim % self.num_heads == 0, "embedding_dim must be divisible by num_heads"
        return self.embedding_dim // self.num_heads

@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000
    eval_interval: int = 100
    checkpoint_interval: int = 200
    seed: int = 42

@dataclass
class SystemConfig:
    device: str = "mps"  # Options: 'mps', 'cpu'
    precision: str = "float32"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    log_dir: str = "logs"
