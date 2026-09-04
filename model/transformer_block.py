#!/usr/bin/env python3
"""
Phase 7 Module: Transformer Block with Pre-LayerNormalization Architecture

Structure:
  x = x + Attention( LayerNorm(x) )
  x = x + FeedForward( LayerNorm(x) )

Why Pre-LayerNormalization (Pre-LN) is used:
--------------------------------------------
Pre-LN applies Layer Normalization directly to the input of each sub-layer (Attention and FFN)
BEFORE the computation, placing the residual addition outside the normalized path.

Benefits over original Post-LN:
1. Unobstructed Gradient Highway: Gradients flow directly through the linear identity skip connection
   (x = x + ...), preventing vanishing or exploding gradients in deep Transformer architectures.
2. Training Stability: Eliminates hyperparameter sensitivity and permits stable optimization from
   step 0 without requiring complex learning rate warmup heuristics.
"""

import torch
import torch.nn as nn
from model.attention import CausalSelfAttention
from model.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    """
    A single Pre-LN Transformer Decoder Block.
    
    Args:
        embedding_dim (int): Model hidden dimension D (e.g. 384).
        num_heads (int): Number of attention heads H (e.g. 6).
        context_length (int): Maximum sequence length N_ctx (e.g. 512).
        ff_multiplier (int): FFN hidden expansion multiplier (e.g. 4 -> 1536).
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        embedding_dim: int = 384,
        num_heads: int = 6,
        context_length: int = 512,
        ff_multiplier: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        # Pre-Attention LayerNorm
        self.ln_1 = nn.LayerNorm(embedding_dim)
        # Causal Multi-Head Self-Attention
        self.attn = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout
        )
        # Pre-FFN LayerNorm
        self.ln_2 = nn.LayerNorm(embedding_dim)
        # Feed-Forward Network
        self.mlp = FeedForward(
            embedding_dim=embedding_dim,
            ff_multiplier=ff_multiplier,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Transformer Block.
        
        Input:  x of shape (B, T, D)
        Output: Tensor of shape (B, T, D)
        """
        # Sub-layer 1: Pre-LN Attention + Residual Connection
        x = x + self.attn(self.ln_1(x))
        
        # Sub-layer 2: Pre-LN Feed-Forward + Residual Connection
        x = x + self.mlp(self.ln_2(x))
        
        return x
