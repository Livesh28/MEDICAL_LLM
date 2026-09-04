#!/usr/bin/env python3
"""
Phase 15 Module: 5-Way Model Comparative Evaluation Benchmark Script
Evaluates 5 candidate models against Gold Benchmark v2 (data/evaluation/venipuncture_gold_eval_v2.json):
1. MedicalTransformerLM (best.pt - v1)
2. MedicalTransformerLM (best_v2.pt - v2)
3. MedicalTransformerLM (best_v3.pt - v3)
4. Llama 3.2 3B without RAG
5. Llama 3.2 3B + RAG
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
from rag.database import LocalVectorDatabase

def query_ollama(prompt: str, model_name: str = "llama3.2:3b", max_tokens: int = 150) -> str:
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

def score_response(generated: str, verified: str, key_terms: list) -> str:
    gen_lower = generated.lower().strip()
    
    if "not available" in gen_lower or "not provided" in gen_lower or "does not modify" in gen_lower:
        return "Correct Refusal"
        
    if not generated.strip():
        return "Unsupported"
        
    matches = sum(1 for term in key_terms if term.lower() in gen_lower)
    match_ratio = matches / max(len(key_terms), 1)
    
    if match_ratio >= 0.6:
        return "Correct"
    elif match_ratio >= 0.2:
        return "Partially Correct"
    else:
        if len(generated.split()) > 5 and matches == 0:
            return "Hallucinated"
        return "Incorrect"

def run_comparison():
    print("=" * 75)
    print("PHASE 15: 5-WAY MODEL COMPARATIVE BENCHMARK EVALUATION")
    print("=" * 75)
    
    gold_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    if not os.path.exists(gold_file):
        raise FileNotFoundError(f"Gold evaluation file not found at {gold_file}")
        
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_dataset = json.load(f)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer_path = "tokenizer/artifacts/tokenizer.json"
    tokenizer = MedicalTokenizer(tokenizer_path)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, embedding_dim=768, num_layers=12, num_heads=12, context_length=512)
    
    db = LocalVectorDatabase()
    db.load("data/rag_db")
    
    # 1. Load Model v1 (best.pt)
    print("\n[+] Loading PyTorch Model v1 (checkpoints/best.pt)...")
    model_v1 = MedicalTransformerLM(cfg).to(device)
    if os.path.exists("checkpoints/best.pt"):
        load_checkpoint("checkpoints/best.pt", model_v1, device=device)
    gen_v1 = MedicalGenerator(model_v1, tokenizer, device)
    
    # 2. Load Model v2 SFT (best_v2.pt)
    print("[+] Loading PyTorch Model v2 SFT (checkpoints/best_v2.pt)...")
    model_v2 = MedicalTransformerLM(cfg).to(device)
    if os.path.exists("checkpoints/best_v2.pt"):
        load_checkpoint("checkpoints/best_v2.pt", model_v2, device=device)
    gen_v2 = MedicalGenerator(model_v2, tokenizer, device)
    
    # 3. Load Model v3 SFT (best_v3.pt)
    print("[+] Loading PyTorch Model v3 SFT (checkpoints/best_v3.pt)...")
    model_v3 = MedicalTransformerLM(cfg).to(device)
    if os.path.exists("checkpoints/best_v3.pt"):
        load_checkpoint("checkpoints/best_v3.pt", model_v3, device=device)
    gen_v3 = MedicalGenerator(model_v3, tokenizer, device)
    
    eval_records = []
    
    stats = {
        "v1_initial":      {"Correct": 0, "Correct Refusal": 0, "Partially Correct": 0, "Incorrect": 0, "Hallucinated": 0, "Unsupported": 0},
        "v2_sft":          {"Correct": 0, "Correct Refusal": 0, "Partially Correct": 0, "Incorrect": 0, "Hallucinated": 0, "Unsupported": 0},
        "v3_sft":          {"Correct": 0, "Correct Refusal": 0, "Partially Correct": 0, "Incorrect": 0, "Hallucinated": 0, "Unsupported": 0},
        "llama3.2_no_rag": {"Correct": 0, "Correct Refusal": 0, "Partially Correct": 0, "Incorrect": 0, "Hallucinated": 0, "Unsupported": 0},
        "llama3.2_rag":    {"Correct": 0, "Correct Refusal": 0, "Partially Correct": 0, "Incorrect": 0, "Hallucinated": 0, "Unsupported": 0}
    }
    
    print("\n[+] Evaluating 5 Candidates across Gold Benchmark Questions...\n")
    
    for idx, item in enumerate(gold_dataset, start=1):
        q_id = item["id"]
        question = item["question"]
        verified = item["verified_answer"]
        topic = item["topic"]
        step = item.get("step")
        
        key_terms = [w.strip(".,();") for w in verified.split() if len(w) > 4 and w.lower() not in ["should", "first", "second", "third", "which", "where", "after", "before"]]
        
        # Candidate 1: Model v1
        p1 = f"Question: {question}\nMedical Answer:"
        ans_v1_raw = gen_v1.generate(prompt=p1, max_new_tokens=80, temperature=0.7, top_k=40, top_p=0.9)
        ans_v1 = ans_v1_raw.split("Medical Answer:")[-1].strip() if "Medical Answer:" in ans_v1_raw else ans_v1_raw.strip()
        s_v1 = score_response(ans_v1, verified, key_terms)
        stats["v1_initial"][s_v1] += 1
        
        # Candidate 2: Model v2
        p2 = f"Instruction: {question}\nMedical Answer:"
        ans_v2_raw = gen_v2.generate(prompt=p2, max_new_tokens=80, temperature=0.3, top_k=20, top_p=0.9)
        ans_v2 = ans_v2_raw.split("Medical Answer:")[-1].strip() if "Medical Answer:" in ans_v2_raw else ans_v2_raw.strip()
        s_v2 = score_response(ans_v2, verified, key_terms)
        stats["v2_sft"][s_v2] += 1
        
        # Candidate 3: Model v3 SFT
        p3 = f"Instruction: {question}\nMedical Answer:"
        ans_v3_raw = gen_v3.generate(prompt=p3, max_new_tokens=80, temperature=0.2, top_k=10, top_p=0.9)
        ans_v3 = ans_v3_raw.split("Medical Answer:")[-1].strip() if "Medical Answer:" in ans_v3_raw else ans_v3_raw.strip()
        s_v3 = score_response(ans_v3, verified, key_terms)
        stats["v3_sft"][s_v3] += 1
        
        # Candidate 4: Llama 3.2 3B without RAG
        p4 = f"Instruction: {question}\nMedical Answer:"
        ans_l_norag = query_ollama(p4, model_name="llama3.2:3b", max_tokens=100)
        s_l_norag = score_response(ans_l_norag, verified, key_terms)
        stats["llama3.2_no_rag"][s_l_norag] += 1
        
        # Candidate 5: Llama 3.2 3B with RAG
        hits = db.search(question, top_k=2)
        snippets = [h[0]["text"][:200] for h in hits] if hits else []
        rag_context = "\n".join(snippets)
        p5 = (
            "You are an expert clinical phlebotomy instructor for a VR venipuncture simulator. "
            "Answer the question accurately using the clinical context below:\n\n"
            f"Clinical Context:\n{rag_context}\n\n"
            f"Question: {question}\n\nMedical Answer:"
        )
        ans_l_rag = query_ollama(p5, model_name="llama3.2:3b", max_tokens=100)
        s_l_rag = score_response(ans_l_rag, verified, key_terms)
        stats["llama3.2_rag"][s_l_rag] += 1
        
        record = {
            "id": q_id,
            "topic": topic,
            "step": step,
            "question": question,
            "verified_answer": verified,
            "scores": {
                "v1_initial": s_v1,
                "v2_sft": s_v2,
                "v3_sft": s_v3,
                "llama3.2_no_rag": s_l_norag,
                "llama3.2_rag": s_l_rag
            }
        }
        eval_records.append(record)
        
        print(f"[{idx:2d}/25] Q: {question[:50]}...")
        print(f"       v1:{s_v1:12s} | v2:{s_v2:12s} | v3:{s_v3:12s} | LlamaNoRAG:{s_l_norag:12s} | LlamaRAG:{s_l_rag}")
        
    total_q = len(gold_dataset)
    
    def calc_metrics(m_dict):
        corr = m_dict["Correct"] + m_dict["Correct Refusal"]
        part = m_dict["Partially Correct"]
        inc = m_dict["Incorrect"]
        hall = m_dict["Hallucinated"]
        return {
            "accuracy_percent": round(corr / total_q * 100, 1),
            "partial_percent": round(part / total_q * 100, 1),
            "incorrect_percent": round(inc / total_q * 100, 1),
            "hallucination_percent": round(hall / total_q * 100, 1),
            "breakdown": m_dict
        }

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_questions": total_q,
        "models": {
            "v1_initial_110m":   calc_metrics(stats["v1_initial"]),
            "v2_sft_110m":       calc_metrics(stats["v2_sft"]),
            "v3_sft_110m":       calc_metrics(stats["v3_sft"]),
            "llama3.2_3b_no_rag": calc_metrics(stats["llama3.2_no_rag"]),
            "llama3.2_3b_rag":    calc_metrics(stats["llama3.2_rag"])
        },
        "detailed_records": eval_records
    }
    
    out_file = "outputs/model_comparison_v3.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
        
    print("\n" + "=" * 75)
    print("FINAL 5-WAY MODEL COMPARISON SUMMARY")
    print("=" * 75)
    for m_name, m_stats in summary_report["models"].items():
        print(f" {m_name:22s} -> Accuracy: {m_stats['accuracy_percent']:5.1f}% | Hallucinations: {m_stats['hallucination_percent']:5.1f}%")
    print(f"\n[+] Saved complete evaluation report -> {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    run_comparison()
