#!/usr/bin/env python3
"""
Three-Model Benchmark Comparison Script
Runs gold-standard evaluation across 6 setups:
  1. OpenBioLLM-8B (No RAG)
  2. Llama 3.2 3B (No RAG)
  3. MedicalTransformerLM 110M (No RAG)
  4. OpenBioLLM-8B + RAG V2
  5. Llama 3.2 3B + RAG V2
  6. MedicalTransformerLM 110M + RAG V2

Saves comprehensive empirical metrics to outputs/three_model_comparison.json.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.model_provider import ModelRouter
from rag.pipeline import MedicalRAGPipeline


def score_response(generated: str, verified: str, key_terms: list) -> str:
    gen_lower = generated.lower().strip()
    
    if any(phrase in gen_lower for phrase in ["don't have enough verified information", "not available", "not provided", "cannot answer"]):
        return "Safe Refusal"
        
    if not generated.strip():
        return "Incorrect"
        
    matches = sum(1 for term in key_terms if term.lower() in gen_lower)
    match_ratio = matches / max(len(key_terms), 1)
    
    if match_ratio >= 0.4:
        return "Correct"
    elif match_ratio >= 0.15:
        return "Partial"
    else:
        if len(generated.split()) > 6 and matches == 0:
            return "Hallucination"
        return "Incorrect"


def run_comparison():
    print("=" * 75, flush=True)
    print("THREE-MODEL COMPREHENSIVE BENCHMARK COMPARISON", flush=True)
    print("Candidates: OpenBioLLM-8B | Llama 3.2 3B | MedicalTransformerLM 110M", flush=True)
    print("Modes:      Direct Generation (No RAG) & Grounded Retrieval (RAG V2)", flush=True)
    print("=" * 75, flush=True)

    gold_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    if not os.path.exists(gold_file):
        print(f"[-] Error: Gold dataset {gold_file} not found!", flush=True)
        sys.exit(1)

    with open(gold_file, "r", encoding="utf-8") as f:
        all_gold = json.load(f)

    # Select representative subset of 10 gold benchmark questions
    gold_questions = all_gold[:10]

    router = ModelRouter()
    rag_pipe = MedicalRAGPipeline()

    setups = [
        {"key": "openbiollm_no_rag", "model": "openbiollm", "rag": False},
        {"key": "llama32_no_rag", "model": "llama32", "rag": False},
        {"key": "medical_transformer_no_rag", "model": "medical_transformer_110m", "rag": False},
        {"key": "openbiollm_rag", "model": "openbiollm", "rag": True},
        {"key": "llama32_rag", "model": "llama32", "rag": True},
        {"key": "medical_transformer_rag", "model": "medical_transformer_110m", "rag": True},
    ]

    results_by_setup = {s["key"]: [] for s in setups}
    stats_by_setup = {
        s["key"]: {"Correct": 0, "Partial": 0, "Incorrect": 0, "Hallucination": 0, "Safe Refusal": 0, "total_latency_ms": 0.0}
        for s in setups
    }

    print(f"\n[+] Running 6 Evaluation Setups across {len(gold_questions)} Benchmark Questions...\n", flush=True)

    for idx, item in enumerate(gold_questions, start=1):
        q_id = item["id"]
        question = item["question"]
        verified = item["verified_answer"]
        key_terms = [w.strip(".,();") for w in verified.split() if len(w) > 4 and w.lower() not in ["should", "first", "second", "third", "which", "where", "after", "before"]]

        print(f"[{idx:2d}/{len(gold_questions)}] Q: {question[:55]}...", flush=True)

        for setup in setups:
            s_key = setup["key"]
            m_key = setup["model"]
            use_rag = setup["rag"]

            t0 = time.time()
            if use_rag:
                res_dict = rag_pipe.answer_question(question=question, model=m_key, max_new_tokens=80, temperature=0.3)
                answer = res_dict.get("answer", "")
                provider_info = {"provider": res_dict.get("provider"), "model": res_dict.get("model")}
            else:
                prompt = f"Instruction: {question}\nMedical Answer:" if m_key == "medical_transformer_110m" else (
                    f"You are a clinical instructor. Answer concisely and accurately:\nQuestion: {question}\nMedical Answer:"
                )
                answer, meta = router.generate(prompt=prompt, model_key=m_key, max_tokens=80, temperature=0.3)
                provider_info = meta

            latency_ms = round((time.time() - t0) * 1000, 2)
            classification = score_response(answer, verified, key_terms)

            stats_by_setup[s_key][classification] += 1
            stats_by_setup[s_key]["total_latency_ms"] += latency_ms

            results_by_setup[s_key].append({
                "question_id": q_id,
                "question": question,
                "verified_answer": verified,
                "generated_answer": answer,
                "classification": classification,
                "latency_ms": latency_ms,
                "provider": provider_info.get("provider"),
                "model": provider_info.get("model")
            })

    total_q = len(gold_questions)
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_questions": total_q,
        "metrics_summary": {}
    }

    print("\n" + "=" * 80, flush=True)
    print("FINAL THREE-MODEL COMPARISON RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"{'SETUP':30s} | {'ACCURACY':10s} | {'HALLUC':10s} | {'REFUSAL':10s} | {'AVG LATENCY':12s}", flush=True)
    print("-" * 80, flush=True)

    for s_key, counts in stats_by_setup.items():
        corr = counts["Correct"]
        part = counts["Partial"]
        inc = counts["Incorrect"]
        hall = counts["Hallucination"]
        ref = counts["Safe Refusal"]
        avg_lat = round(counts["total_latency_ms"] / max(total_q, 1), 2)
        acc_pct = round((corr + part) / max(total_q, 1) * 100, 1)
        hall_pct = round(hall / max(total_q, 1) * 100, 1)
        ref_pct = round(ref / max(total_q, 1) * 100, 1)

        summary_report["metrics_summary"][s_key] = {
            "correct_count": corr,
            "partial_count": part,
            "incorrect_count": inc,
            "hallucination_count": hall,
            "safe_refusal_count": ref,
            "accuracy_percent": acc_pct,
            "hallucination_percent": hall_pct,
            "safe_refusal_percent": ref_pct,
            "average_latency_ms": avg_lat,
            "raw_breakdown": counts
        }

        print(f"{s_key:30s} | {acc_pct:9.1f}% | {hall_pct:9.1f}% | {ref_pct:9.1f}% | {avg_lat:9.2f} ms", flush=True)

    summary_report["detailed_runs"] = results_by_setup
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "three_model_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("=" * 80, flush=True)
    print(f"[✓] Three-Model Comparison saved successfully to {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_comparison()
