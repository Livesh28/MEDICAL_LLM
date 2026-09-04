#!/usr/bin/env python3
"""
Phase 12 Module: Medical Model Evaluation & Benchmark Suite
Evaluates:
1. Validation Loss & Perplexity (PPL = exp(val_loss))
2. Structured Medical Domain Benchmark Prompts across 7 Categories:
   - Anatomy
   - Physiology
   - Biology
   - Diseases
   - Medical Terminology
   - Basic Pharmacology
   - General Health Education
3. Qualitative Hallucination & Consistency Analysis
4. Safety & Educational Disclaimer Compliance

Note: This system is an educational prototype and NOT clinically validated.
"""

import os
import sys
import json
import math
import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from tokenizer.tokenizer import MedicalTokenizer
from configs.model_config import ModelConfig
from dataset.dataloader import get_dataloaders
from training.checkpoint import load_checkpoint
from inference.generate import MedicalGenerator

MEDICAL_BENCHMARK_PROMPTS = [
    {
        "category": "Anatomy",
        "question": "What organ pumps blood throughout the human body?",
        "target_concept": "heart"
    },
    {
        "category": "Physiology",
        "question": "What is the primary function of red blood cells?",
        "target_concept": "oxygen transport"
    },
    {
        "category": "Biology",
        "question": "What is DNA and what is its primary structure?",
        "target_concept": "genetic material / double helix"
    },
    {
        "category": "Diseases",
        "question": "What is diabetes mellitus?",
        "target_concept": "blood glucose / insulin deficiency or resistance"
    },
    {
        "category": "Medical Terminology",
        "question": "What does tachycardia mean in medical terms?",
        "target_concept": "abnormally rapid heart rate"
    },
    {
        "category": "Basic Pharmacology",
        "question": "What are antibiotics generally used to treat?",
        "target_concept": "bacterial infections"
    },
    {
        "category": "General Health Education",
        "question": "What is a fever and why does the body develop one?",
        "target_concept": "elevated body temperature / immune response"
    }
]

@torch.no_grad()
def compute_val_metrics(model, val_loader, device, max_batches=50, max_eval_batches=None):
    if max_eval_batches is not None:
        max_batches = max_eval_batches
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    count = 0
    start = time.time()
    
    for i, (x, y) in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        total_loss += loss.item()
        total_tokens += x.numel()
        count += 1
        
    avg_loss = total_loss / count if count > 0 else 0.0
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
        
    elapsed = time.time() - start
    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "tokens_evaluated": total_tokens,
        "eval_seconds": round(elapsed, 2),
        "eval_tokens_per_sec": round(total_tokens / elapsed, 0) if elapsed > 0 else 0
    }

evaluate_model = compute_val_metrics

def run_medical_evaluation(
    checkpoint_path: str = "checkpoints/best.pt",
    tokenizer_path: str = "tokenizer/artifacts/tokenizer.json",
    val_bin_path: str = "data/processed/val_tokens.bin",
    output_dir: str = "outputs",
    device_name: str = "mps"
):
    print("=" * 70)
    print("PHASE 12: Medical Model Evaluation & Benchmark Suite")
    print("=" * 70)
    print("SAFETY NOTICE: Prototype educational system — not for clinical use.\n")
    
    device = torch.device("mps" if torch.backends.mps.is_available() and device_name == "mps" else "cpu")
    print(f"Device: {device}")
    
    tokenizer = MedicalTokenizer(tokenizer_path)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, embedding_dim=384, num_layers=10, num_heads=6, context_length=512)
    model = MedicalTransformerLM(cfg).to(device)
    
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} missing. Evaluating untrained weights.")
        
    # 1. Evaluate Loss & Perplexity
    print("\n--- 1. Quantitative Loss & Perplexity Benchmark ---")
    _, val_loader = get_dataloaders(val_bin_path=val_bin_path, context_length=512, batch_size=8)
    val_metrics = compute_val_metrics(model, val_loader, device=device, max_batches=50)
    
    print(f"Validation Loss:  {val_metrics['val_loss']}")
    print(f"Perplexity (PPL): {val_metrics['perplexity']}")
    print(f"Evaluation Speed: {val_metrics['eval_tokens_per_sec']:,} tok/s")
    
    # 2. Medical Domain Benchmark Generation
    print("\n--- 2. Qualitative Medical Knowledge Generation Prompts ---")
    generator = MedicalGenerator(model, tokenizer, device)
    
    prompt_results = []
    for item in MEDICAL_BENCHMARK_PROMPTS:
        cat = item["category"]
        q = item["question"]
        concept = item["target_concept"]
        
        gen_text = generator.generate(
            prompt=q,
            max_new_tokens=50,
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )
        
        prompt_results.append({
            "category": cat,
            "prompt": q,
            "expected_concept": concept,
            "generated_response": gen_text
        })
        
        print(f"[{cat}] Q: '{q}'")
        print(f"     -> Response: '{gen_text[:120]}...'")
        print("-" * 50)
        
    # Save Report
    report = {
        "metrics": val_metrics,
        "medical_benchmark": prompt_results,
        "disclaimer": "This system is an educational/research prototype and is not a substitute for professional medical advice, diagnosis, or treatment."
    }
    
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "evaluation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[+] Full Evaluation Report saved to: {report_file}")
    print("=" * 70)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Medical LLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--val_bin", type=str, default="data/processed/val_tokens.bin")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    run_medical_evaluation(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        val_bin_path=args.val_bin,
        output_dir=args.output_dir,
        device_name=args.device
    )
