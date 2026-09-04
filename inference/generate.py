#!/usr/bin/env python3
"""
Phase 11 Module: Autoregressive Text Generation System
Implements local text generation using top-k, top-p (nucleus sampling), temperature scaling,
greedy decoding option, and EOS token stopping on Apple Silicon MPS.

Disclaimer: Educational/Research prototype only — not for clinical use.
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
from typing import Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from tokenizer.tokenizer import MedicalTokenizer
from configs.model_config import ModelConfig
from training.checkpoint import load_checkpoint

class MedicalGenerator:
    """
    Autoregressive Text Generator for Local Medical Transformer LLM.
    """
    def __init__(
        self,
        model: MedicalTransformerLM,
        tokenizer: MedicalTokenizer,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval().to(device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        stop_on_eos: bool = True
    ) -> str:
        """
        Generates text autoregressively starting from prompt.
        
        Args:
            prompt: Text prompt string.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (>0.0 for sampling, 0.0 for greedy).
            top_k: Top-k vocabulary truncation threshold.
            top_p: Top-p (nucleus) cumulative probability threshold.
            stop_on_eos: Whether to stop when EOS / EOT token is emitted.
            
        Returns:
            Generated text string (prompt + generated tokens).
        """
        # Encode prompt
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [self.tokenizer.bos_id or 0]
            
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        stop_ids = set()
        if self.tokenizer.eos_id is not None:
            stop_ids.add(self.tokenizer.eos_id)
        if self.tokenizer.eot_id is not None:
            stop_ids.add(self.tokenizer.eot_id)
            
        context_len = self.model.config.context_length

        for _ in range(max_new_tokens):
            # Crop context if length exceeds max model context_length
            x_cond = x if x.size(1) <= context_len else x[:, -context_len:]
            
            # Forward pass: get logits for final token
            logits, _ = self.model(x_cond)
            logits = logits[:, -1, :]  # Shape: (1, V)
            
            if temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature scaling
                logits = logits / max(temperature, 1e-5)
                
                # Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                    
                # Top-P (Nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices right to keep first token exceeding top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("Inf")
                    
                # Softmax probabilities & sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            # Append next token to context
            x = torch.cat((x, next_token), dim=1)
            
            # EOS stopping check
            token_item = next_token.item()
            if stop_on_eos and token_item in stop_ids:
                break
                
        # Decode token sequence
        generated_ids = x[0].tolist()
        text = self.tokenizer.decode(generated_ids)
        
        # Clean template headers and MCQ option dicts if present
        import re
        text = re.sub(r'Answer this question truthfully\n?', '', text)
        text = re.sub(r'Instruction: [^\n]+\n?', '', text)
        text = re.sub(r'Context/Input:\s*', '', text)
        text = re.sub(r'Medical Details:\s*', '', text)
        text = re.sub(r"['\"][A-E]['\"]:\s*['\"][^'\"]*['\"]", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def main():
    parser = argparse.ArgumentParser(description="Autoregressive Text Generation for Medical LLM")
    parser.add_argument("--prompt", type=str, default="What is the primary function of the heart?", help="Input prompt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    print("=" * 70)
    print("LOCAL MEDICAL LLM — AUTOREGRESSIVE TEXT GENERATION")
    print("=" * 70)
    print("[!] DISCLAIMER: This system is an educational/research prototype.")
    print("    It is NOT a substitute for professional medical advice or diagnosis.")
    print("-" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() and args.device == "mps" else "cpu")
    print(f"Device:               {device}")
    print(f"Loading Tokenizer:    {args.tokenizer_path}")
    
    tokenizer = MedicalTokenizer(args.tokenizer_path)
    
    # Load Model Architecture & Checkpoint Weights
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=512
    )
    model = MedicalTransformerLM(cfg)
    
    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device)
    else:
        print(f"[!] Warning: Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    generator = MedicalGenerator(model, tokenizer, device)

    print(f"\nPrompt:              '{args.prompt}'")
    print(f"Sampling Parameters: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("-" * 70)
    print("Generating Response...\n")

    output_text = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print("--- Generated Text ---")
    print(output_text)
    print("=" * 70)

if __name__ == "__main__":
    main()
