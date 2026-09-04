#!/usr/bin/env python3
"""
Phase 2 Module: Controlled Evaluation & Baseline Benchmark Script
Evaluates 25 core venipuncture questions across MedicalTransformerLM (110M) and Llama 3.2 3B (Ollama).
Outputs detailed benchmark responses to outputs/baseline_eval_results.json.
"""

import os
import sys
import json
import time
import requests
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from tokenizer.tokenizer import MedicalTokenizer
from configs.model_config import ModelConfig
from training.checkpoint import load_checkpoint
from inference.generate import MedicalGenerator

BENCHMARK_QUESTIONS = [
    # Basic Venipuncture Knowledge & Sequence
    {"id": "q01", "topic": "procedural_sequence", "step": 1, "question": "What is the very first step in the 16-step VR venipuncture procedure?"},
    {"id": "q02", "topic": "patient_preparation", "step": 2, "question": "How should you correctly identify the patient before beginning venipuncture?"},
    {"id": "q03", "topic": "equipment", "step": 3, "question": "What equipment must be verified and assembled before performing venipuncture?"},
    {"id": "q04", "topic": "tourniquet", "step": 4, "question": "Where and how tight should the tourniquet be applied on the patient arm?"},
    {"id": "q05", "topic": "tourniquet", "step": 4, "question": "What is the maximum amount of time a tourniquet can safely remain tied on the arm?"},
    
    # Site Selection & Skin Prep
    {"id": "q06", "topic": "site_selection", "step": 5, "question": "Which vein is the first choice for venipuncture in the antecubital fossa and why?"},
    {"id": "q07", "topic": "site_selection", "step": 5, "question": "How do you properly palpate a vein to distinguish it from an artery?"},
    {"id": "q08", "topic": "skin_preparation", "step": 6, "question": "What antiseptic is used for routine skin cleaning and how should it be applied?"},
    {"id": "q09", "topic": "skin_preparation", "step": 6, "question": "Why must the disinfected skin be allowed to air dry for 30 seconds before needle insertion?"},
    
    # Needle Insertion & Blood Collection
    {"id": "q10", "topic": "needle_insertion", "step": 7, "question": "What is the correct angle of entry and bevel orientation when inserting the needle into the vein?"},
    {"id": "q11", "topic": "needle_insertion", "step": 7, "question": "How do you anchor the vein prior to inserting the needle?"},
    {"id": "q12", "topic": "order_of_draw", "step": 8, "question": "What is the correct CLSI order of draw for blood collection tubes?"},
    {"id": "q13", "topic": "order_of_draw", "step": 8, "question": "Why is the order of draw critical during blood collection?"},
    {"id": "q14", "topic": "tube_handling", "step": 9, "question": "Why and how many times should blood collection tubes containing additives be inverted?"},
    
    # Procedural Completion & Safety
    {"id": "q15", "topic": "tourniquet_release", "step": 10, "question": "When should the tourniquet be released during the venipuncture procedure?"},
    {"id": "q16", "topic": "needle_withdrawal", "step": 11, "question": "How should the needle be withdrawn and when is gauze pressure applied?"},
    {"id": "q17", "topic": "safety_disposal", "step": 12, "question": "When and how should the safety device on the needle be activated?"},
    {"id": "q18", "topic": "safety_disposal", "step": 13, "question": "Where must the used needle assembly be disposed of immediately after activation?"},
    {"id": "q19", "topic": "specimen_labeling", "step": 14, "question": "When and where should blood collection tubes be labeled with patient information?"},
    {"id": "q20", "topic": "post_procedure", "step": 15, "question": "What instructions should be given to the patient regarding post-venipuncture care?"},
    
    # Common Mistakes & VR Trainee Assistance
    {"id": "q21", "topic": "common_mistakes", "step": 4, "question": "What happens if a tourniquet is left tied on the arm for longer than 1 minute?"},
    {"id": "q22", "topic": "common_mistakes", "step": 7, "question": "What is hemolysis and what procedural mistakes cause it?"},
    {"id": "q23", "topic": "common_mistakes", "step": 7, "question": "Why is probing or repuncturing with the needle prohibited if blood flow is not established?"},
    {"id": "q24", "topic": "vr_workflow", "step": 8, "question": "In the VR venipuncture simulator, why was tube insertion marked wrong if a blue tube was inserted before a red tube?"},
    {"id": "q25", "topic": "vr_workflow", "step": 6, "question": "In the VR trainer, why does touching the palpated vein after alcohol cleaning trigger a step error?"}
]

def query_ollama(prompt: str, model_name: str = "llama3.2:3b", max_tokens: int = 150) -> str:
    """Queries local Ollama instance for benchmark response."""
    try:
        res = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens}
            },
            timeout=10
        )
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except Exception as e:
        return f"[Ollama Error: {e}]"
    return "[Ollama Unavailable]"

def run_evaluation():
    print("=" * 70)
    print("PHASE 2: CONTROLLED BENCHMARK EVALUATION (25 VENIPUNCTURE QUESTIONS)")
    print("=" * 70)
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Loading MedicalTransformerLM (110M) on device: {device}...")
    
    tokenizer_path = "tokenizer/artifacts/tokenizer.json"
    checkpoint_path = "checkpoints/best.pt"
    
    tokenizer = MedicalTokenizer(tokenizer_path)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, embedding_dim=768, num_layers=12, num_heads=12, context_length=512)
    model = MedicalTransformerLM(cfg)
    
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, device=device)
        print(f"[+] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running on initialized weights.")
        
    generator = MedicalGenerator(model, tokenizer, device)
    
    results = []
    print("\n[+] Executing side-by-side model evaluation...")
    
    for idx, item in enumerate(BENCHMARK_QUESTIONS, start=1):
        q_id = item["id"]
        question = item["question"]
        topic = item["topic"]
        step = item["step"]
        
        print(f" [{idx:2d}/25] Question ID: {q_id} (Topic: {topic}, Step: {step})")
        print(f"      Q: {question}")
        
        # 1. Custom 110M PyTorch Model Response
        prompt_str = f"Question: {question}\nMedical Answer:"
        t0 = time.time()
        pytorch_raw = generator.generate(prompt=prompt_str, max_new_tokens=100, temperature=0.7, top_k=40, top_p=0.9)
        t_pytorch = time.time() - t0
        
        if prompt_str in pytorch_raw:
            pytorch_ans = pytorch_raw.split(prompt_str)[-1].strip()
        else:
            pytorch_ans = pytorch_raw.strip()
            
        # 2. Ollama Llama 3.2 3B Response
        ollama_prompt = (
            "You are an expert clinical phlebotomy instructor for a VR venipuncture simulator. "
            "Provide a concise, direct, medically accurate answer to the following question:\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        t0 = time.time()
        ollama_ans = query_ollama(ollama_prompt, model_name="llama3.2:3b", max_tokens=120)
        t_ollama = time.time() - t0
        
        record = {
            "id": q_id,
            "topic": topic,
            "step": step,
            "question": question,
            "pytorch_110m": {
                "answer": pytorch_ans,
                "latency_sec": round(t_pytorch, 3),
                "settings": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_new_tokens": 100}
            },
            "llama3.2_3b": {
                "answer": ollama_ans,
                "latency_sec": round(t_ollama, 3),
                "settings": {"temperature": 0.3, "max_tokens": 120}
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        results.append(record)
        
    eval_file = os.path.join(output_dir, "baseline_eval_results.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 70)
    print(f"[+] Evaluation Complete! Saved 25 benchmark comparison records to: {eval_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
