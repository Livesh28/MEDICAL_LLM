#!/usr/bin/env python3
"""
Phase 5 Module: Transformer Embeddings
Implements token embeddings and learned positional embeddings.

Input Tensor Shape:  (batch_size, sequence_length)  -> (B, T)
Output Tensor Shape: (batch_size, sequence_length, embedding_dimension) -> (B, T, D)
"""

import torch
import torch.nn as nn
from typing import Optional

class TransformerEmbeddings(nn.Module):
    """
    Combines learned token embeddings with learned positional embeddings.
    
    Args:
        vocab_size (int): Size of tokenizer vocabulary V.
        embedding_dim (int): Model hidden embedding dimension D.
        context_length (int): Maximum sequence length N_ctx.
        dropout (float): Dropout probability applied after embedding sum.
    """
    def __init__(
        self,
        vocab_size: int = 16000,
        embedding_dim: int = 384,
        context_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        
        # Token Embedding lookup table: maps token IDs to D-dimensional vectors
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Learned Positional Embedding lookup table: maps position indices [0..T-1] to D-dimensional vectors
        self.position_embeddings = nn.Embedding(context_length, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for embedding layer.
        
        Input:
            x: LongTensor of shape (B, T) containing token IDs
               where B = batch_size, T = sequence_length (T <= context_length)
               
        Output:
            Tensor of shape (B, T, D) containing combined token + positional embeddings
        """
        batch_size, seq_len = x.shape
        assert seq_len <= self.context_length, (
            f"Sequence length {seq_len} exceeds max model context_length {self.context_length}."
        )
        
        # Generate position indices [0, 1, 2, ..., seq_len - 1] on same device as x
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)  # Shape: (T,)
        
        # Token Embeddings: (B, T) -> (B, T, D)
        tok_emb = self.token_embeddings(x)
        
        # Positional Embeddings: (T,) -> (T, D) -> broadcast to (B, T, D)
        pos_emb = self.position_embeddings(positions)
        
        # Sum token & positional embeddings + apply dropout
        embeddings = tok_emb + pos_emb  # Shape: (B, T, D)
        return self.dropout(embeddings)
