#!/usr/bin/env python3
"""
Phase 6 Module: Causal Multi-Head Self-Attention
Implements multi-head causal self-attention from scratch using PyTorch modules.

Requirements:
- Query (Q), Key (K), Value (V) projections
- Multi-head splitting: (B, T, D) -> (B, H, T, d_k)
- Scaled Dot-Product Attention: Scores = (Q @ K.T) / sqrt(d_k)
- Causal Masking: prevents attention to future tokens (upper triangle masked to -inf)
- Softmax & Attention Dropout
- Value aggregation: Output = Attention @ V -> (B, H, T, d_k)
- Re-combining heads & Output projection: (B, T, D)
"""

import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention Layer.
    
    Args:
        embedding_dim (int): Model hidden dimension D (e.g. 384).
        num_heads (int): Number of parallel attention heads H (e.g. 6).
        context_length (int): Max sequence length N_ctx (e.g. 512).
        dropout (float): Attention and residual projection dropout rate.
    """
    def __init__(
        self,
        embedding_dim: int = 384,
        num_heads: int = 6,
        context_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        )
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads  # d_k
        self.context_length = context_length
        
        # Combined Q, K, V linear projection for memory efficiency: D -> 3 * D
        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim)
        
        # Output projection linear layer: D -> D
        self.c_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Registered lower-triangular causal mask buffer (1, 1, N_ctx, N_ctx)
        # Position (i, j) is 1 if j <= i (past/present allowed), 0 if j > i (future blocked)
        causal_mask = torch.tril(torch.ones(context_length, context_length)).view(
            1, 1, context_length, context_length
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal multi-head self-attention.
        
        Input:
            x: Tensor of shape (B, T, D)
               where B = batch_size, T = sequence_length, D = embedding_dim
               
        Output:
            Tensor of shape (B, T, D)
        """
        B, T, D = x.size()
        assert D == self.embedding_dim, f"Expected input embedding dim {self.embedding_dim}, got {D}"
        assert T <= self.context_length, f"Sequence length {T} exceeds max context length {self.context_length}"
        
        # 1. Project input to Query, Key, Value vectors: (B, T, D) -> (B, T, 3*D)
        qkv = self.c_attn(x)
        
        # Split into Q, K, V: each is (B, T, D)
        q, k, v = qkv.split(self.embedding_dim, dim=2)
        
        # 2. Reshape and transpose for multi-head attention:
        # (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention:
        # Scores = (Q @ K^T) / sqrt(d_k)
        # (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. Apply Causal Mask:
        # Mask out upper triangle (future tokens) with -inf
        mask = self.causal_mask[:, :, :T, :T]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        # 5. Softmax to obtain attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Shape: (B, H, T, T)
        
        # Attention Dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # 6. Aggregate Values:
        # (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        out = torch.matmul(attn_weights, v)
        
        # 7. Re-combine heads:
        # Transpose back: (B, H, T, d_k) -> (B, T, H, d_k)
        # Flatten heads: (B, T, H * d_k) = (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        # 8. Output projection & residual dropout
        out = self.c_proj(out)
        return self.resid_dropout(out)
