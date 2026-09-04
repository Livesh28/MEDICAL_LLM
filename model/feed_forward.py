#!/usr/bin/env python3
"""
Phase 7 Module: Transformer Feed-Forward Network (FFN / MLP)
Implements position-wise feed-forward network with GELU activation and dropout.

Structure:
  Linear(D -> 4*D) -> GELU -> Dropout -> Linear(4*D -> D) -> Dropout

Input Shape:  (B, T, D)
Output Shape: (B, T, D)
"""

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP).
    
    Args:
        embedding_dim (int): Model hidden dimension D (e.g. 384).
        ff_multiplier (int): Expansion factor for internal hidden layer (default: 4 -> 1536).
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        embedding_dim: int = 384,
        ff_multiplier: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_multiplier * embedding_dim  # 4 * 384 = 1536
        
        # Up-projection: D -> 4*D
        self.c_fc = nn.Linear(embedding_dim, self.ff_dim)
        # Activation function
        self.act = nn.GELU()
        # Down-projection: 4*D -> D
        self.c_proj = nn.Linear(self.ff_dim, embedding_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Feed-Forward Network.
        
        Input:  x of shape (B, T, D)
        Output: Tensor of shape (B, T, D)
        """
        # Up-project & activate: (B, T, D) -> (B, T, 4*D)
        h = self.act(self.c_fc(x))
        # Down-project & dropout: (B, T, 4*D) -> (B, T, D)
        h = self.c_proj(h)
        return self.dropout(h)
