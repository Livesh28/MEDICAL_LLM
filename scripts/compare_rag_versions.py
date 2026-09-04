#!/usr/bin/env python3
"""
Phases 11 & 13 Module: RAG V2 Controlled 5-Condition Benchmark Script
Evaluates 5 controlled pipeline conditions on data/evaluation/venipuncture_gold_eval_v2.json:
- Condition A: Llama 3.2 3B Standalone
- Condition B: Llama 3.2 3B + TF-IDF RAG
- Condition C: Llama 3.2 3B + BM25 Baseline RAG
- Condition D: Llama 3.2 3B + BM25 + Grounded Prompt v2
- Condition E: Llama 3.2 3B + RAG V2 (BM25 Hybrid + Prompt v3 + Grounding Checker)

Outputs:
- outputs/rag_v2_model_comparison.json
- outputs/rag_v2_failure_analysis.json
"""

import os
import sys
import json
import requests
import math
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase
from rag.retriever_v2 import MetadataAwareHybridRetriever
from scripts.validate_grounding import validate_answer_grounding

def query_ollama(prompt: str, model_name: str = "llama3.2:3b", max_tokens: int = 120) -> str:
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

def score_response(generated: str, verified: str, is_unsupported: bool = False) -> str:
    gen_lower = generated.lower().strip()
    if is_unsupported:
        if "not provided" in gen_lower or "not available" in gen_lower or "don't have" in gen_lower or "cannot" in gen_lower:
            return "Correct Refusal"
        return "Hallucinated"
        
    key_terms = [w.lower() for w in verified.split() if len(w) > 4][:5]
    matches = sum(1 for k in key_terms if k in gen_lower)
    match_ratio = matches / max(len(key_terms), 1)
    
    if match_ratio >= 0.6:
        return "Correct"
    elif match_ratio >= 0.2:
        return "Partially Correct"
    else:
        if len(generated.split()) > 5 and matches == 0:
            return "Hallucinated"
        return "Incorrect"

def run_controlled_benchmark():
    print("=" * 75)
    print("PHASES 11 & 13: RAG V2 CONTROLLED 5-CONDITION BENCHMARK")
    print("=" * 75)
    
    gold_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_dataset = json.load(f)
        
    db = LocalVectorDatabase()
    db.load("data/rag_db")
    hybrid_retriever = MetadataAwareHybridRetriever(db.chunks)
    
    conditions = ["cond_A_standalone", "cond_B_tfidf_rag", "cond_C_bm25_rag", "cond_D_bm25_prompt_v2", "cond_E_rag_v2_prompt_v3_checker"]
    stats = {c: {"Correct": 0, "Correct Refusal": 0, "Partially Correct": 0, "Incorrect": 0, "Hallucinated": 0} for c in conditions}
    
    failure_counts = {
        "retrieval_failure": 0,
        "ranking_failure": 0,
        "generation_failure": 0,
        "unsupported_refusal": 0,
        "success": 0
    }
    
    for idx, item in enumerate(gold_dataset, start=1):
        q = item["question"]
        gold = item["verified_answer"]
        exp_step = item.get("step")
        is_unsup = item.get("topic") == "unsupported_safeguard"
        
        # Cond A
        p_A = f"Instruction: {q}\nMedical Answer:"
        ans_A = query_ollama(p_A)
        stats["cond_A_standalone"][score_response(ans_A, gold, is_unsup)] += 1
        
        # Cond B
        hits_b = db.search(q, top_k=2)
        ctx_b = "\n".join([h[0]["text"][:200] for h in hits_b]) if hits_b else ""
        p_B = f"Context:\n{ctx_b}\n\nQuestion: {q}\nMedical Answer:"
        ans_B = query_ollama(p_B)
        stats["cond_B_tfidf_rag"][score_response(ans_B, gold, is_unsup)] += 1
        
        # Cond C
        scored_c = hybrid_retriever.search(q, top_k=2, use_normalization=False, use_metadata=False)
        ctx_c = "\n".join([h[0]["text"][:200] for h in scored_c]) if scored_c else ""
        p_C = f"Context:\n{ctx_c}\n\nQuestion: {q}\nMedical Answer:"
        ans_C = query_ollama(p_C)
        stats["cond_C_bm25_rag"][score_response(ans_C, gold, is_unsup)] += 1
        
        # Cond D
        p_D = (
            "You are an expert clinical phlebotomy instructor for a VR venipuncture simulator. "
            "Ground your answer strictly in the context below. If asked for patient details, state they are not provided.\n\n"
            f"Context:\n{ctx_c}\n\nQuestion: {q}\nMedical Answer:"
        )
        ans_D = query_ollama(p_D)
        stats["cond_D_bm25_prompt_v2"][score_response(ans_D, gold, is_unsup)] += 1
        
        # Cond E (RAG V2: Hybrid Search + Prompt v3 + Grounding Checker)
        scored_e = hybrid_retriever.search(q, top_k=2, current_step=exp_step, use_normalization=True, use_metadata=True)
        chunks_e = [h[0] for h in scored_e]
        ctx_e = "\n".join([c["text"][:200] for c in chunks_e]) if chunks_e else ""
        p_E = (
            "You are an expert clinical phlebotomy instructor for a Medical VR Venipuncture Training Simulation.\n"
            "Ground your answer strictly in the clinical evidence below. "
            "If asked for patient-specific details (blood pressure, medications), state that the information is not provided.\n\n"
            f"Retrieved Clinical Evidence:\n{ctx_e}\n\n"
            f"Trainee Question: {q}\nMedical Answer:"
        )
        ans_E_raw = query_ollama(p_E)
        ans_E, _, _ = validate_answer_grounding(ans_E_raw, chunks_e, q)
        score_e = score_response(ans_E, gold, is_unsup)
        stats["cond_E_rag_v2_prompt_v3_checker"][score_e] += 1
        
        if score_e in ["Correct", "Correct Refusal"]:
            failure_counts["success"] += 1
        elif is_unsup:
            failure_counts["unsupported_refusal"] += 1
        else:
            failure_counts["generation_failure"] += 1
            
        print(f"[{idx:2d}/25] Cond E Score: {score_e:18s} | Q: {q[:45]}...")

    total = len(gold_dataset)
    report = {"total_questions": total, "conditions": {}}
    
    for c_name, m_dict in stats.items():
        corr = m_dict["Correct"] + m_dict["Correct Refusal"]
        part = m_dict["Partially Correct"]
        inc = m_dict["Incorrect"]
        hall = m_dict["Hallucinated"]
        report["conditions"][c_name] = {
            "accuracy_percent": round(corr / total * 100, 1),
            "partially_correct_percent": round(part / total * 100, 1),
            "incorrect_percent": round(inc / total * 100, 1),
            "hallucination_percent": round(hall / total * 100, 1),
            "total_guidance_coverage": round((corr + part) / total * 100, 1),
            "breakdown": m_dict
        }
        
    out_file = "outputs/rag_v2_model_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    out_fail = "outputs/rag_v2_failure_analysis.json"
    with open(out_fail, "w", encoding="utf-8") as f:
        json.dump({"total_evaluated": total, "breakdown": failure_counts}, f, indent=2)
        
    print("\n" + "=" * 75)
    print("FINAL RAG V2 5-CONDITION CONTROLLED BENCHMARK SUMMARY")
    print("=" * 75)
    for c_name, m_stats in report["conditions"].items():
        print(
            f" {c_name:32s} -> Acc: {m_stats['accuracy_percent']:5.1f}% | "
            f"Partial: {m_stats['partially_correct_percent']:5.1f}% | "
            f"Hallucination: {m_stats['hallucination_percent']:5.1f}% | "
            f"Coverage: {m_stats['total_guidance_coverage']:5.1f}%"
        )
    print(f"\n[+] Saved RAG V2 model comparison -> {out_file}")
    print(f"[+] Saved RAG V2 failure analysis -> {out_fail}")
    print("=" * 75)

if __name__ == "__main__":
    run_controlled_benchmark()
