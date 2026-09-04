#!/usr/bin/env python3
"""
Phase 8 Module: Full Medical Decoder-Only Transformer Language Model (~30M Parameters)
Combines Token/Positional Embeddings, stacked Pre-LN Transformer Blocks, Final LayerNorm,
and Vocabulary Projection LM Head.

Architecture Flow:
  Token IDs (B, T)
        │
        ▼
  Token + Positional Embeddings (B, T, D)
        │
        ▼
  Stacked Transformer Blocks x L (Pre-LN, MHA, FFN, Residual)
        │
        ▼
  Final LayerNorm (B, T, D)
        │
        ▼
  LM Head Linear Projection (B, T, V)
        │
        ▼
  Logits (B, T, V)  [-> CrossEntropyLoss if targets provided]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.embeddings import TransformerEmbeddings
from model.transformer_block import TransformerBlock
from configs.model_config import ModelConfig

class MedicalTransformerLM(nn.Module):
    """
    Decoder-Only Transformer Language Model for Medical Text Generation.
    Default Config (~30.25M Parameters):
      - Vocab Size V: 16,000
      - Embedding Dim D: 384
      - Layers L: 10
      - Heads H: 6 (Head Dim: 64)
      - FFN Dim: 1,536 (4 * D)
      - Context Length N_ctx: 512
    """
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # 1. Embeddings Layer
        self.embeddings = TransformerEmbeddings(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            context_length=config.context_length,
            dropout=config.dropout
        )

        # 2. Stacked Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                context_length=config.context_length,
                ff_multiplier=config.ff_multiplier,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # 3. Final Layer Normalization
        self.ln_f = nn.LayerNorm(config.embedding_dim)

        # 4. Vocabulary Linear Projection (LM Head)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

        # Optional Weight Tying (Token Embeddings & LM Head) can be enabled if desired,
        # but untied gives precise ~30.25M params for 10 layers & 16K vocab.
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization for Transformer modules.
        Linear weights ~ N(0, 0.02), Embeddings ~ N(0, 0.02), Biases = 0.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Calculate total trainable parameters.
        """
        if non_embedding:
            pos_emb_params = self.embeddings.position_embeddings.weight.numel()
            tok_emb_params = self.embeddings.token_embeddings.weight.numel()
            total = sum(p.numel() for p in self.parameters())
            return total - (pos_emb_params + tok_emb_params)
        return sum(p.numel() for p in self.parameters())

    def print_param_summary(self):
        """
        Prints detailed parameter breakdown table.
        """
        emb_params = sum(p.numel() for p in self.embeddings.parameters())
        block_params = sum(p.numel() for p in self.blocks.parameters())
        ln_f_params = sum(p.numel() for p in self.ln_f.parameters())
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        total_params = self.get_num_params()

        print("-" * 60)
        print("EXACT MODEL PARAMETER COUNT SUMMARY")
        print("-" * 60)
        print(f"Embedding Layers:        {emb_params:,} ({emb_params/total_params*100:.1f}%)")
        print(f"Transformer Blocks ({self.config.num_layers}x): {block_params:,} ({block_params/total_params*100:.1f}%)")
        print(f"Final LayerNorm:         {ln_f_params:,}")
        print(f"LM Head Projection:      {lm_head_params:,} ({lm_head_params/total_params*100:.1f}%)")
        print("-" * 60)
        print(f"TOTAL PARAMETER COUNT:   {total_params:,} ({total_params/1e6:.2f} Million Parameters)")
        print("-" * 60)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Language Model.
        
        Args:
            idx: LongTensor of shape (B, T) containing token IDs.
            targets: Optional LongTensor of shape (B, T) containing target token IDs.
            
        Returns:
            Tuple of (logits, loss):
              - logits: FloatTensor of shape (B, T, V)
              - loss: Scalar CrossEntropyLoss if targets provided, else None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.context_length, (
            f"Cannot forward sequence length {t}, max context length is {self.config.context_length}"
        )

        # 1. Embeddings: (B, T) -> (B, T, D)
        x = self.embeddings(idx)

        # 2. Pass through Transformer Blocks: (B, T, D) -> (B, T, D)
        for block in self.blocks:
            x = block(x)

        # 3. Final LayerNorm: (B, T, D) -> (B, T, D)
        x = self.ln_f(x)

        # 4. LM Head Projection: (B, T, D) -> (B, T, V)
        logits = self.lm_head(x)

        # 5. Loss Calculation (Autoregressive Next-Token Cross-Entropy Loss)
        loss = None
        if targets is not None:
            # Flatten logits to (B*T, V) and targets to (B*T,)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)

        return logits, loss
