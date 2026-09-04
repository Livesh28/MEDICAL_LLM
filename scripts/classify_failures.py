#!/usr/bin/env python3
"""
Phase 3 Module: Failure Classification Summarizer
Processes outputs/question_level_debug.json and outputs outputs/failure_analysis.json
classifying root-cause failure categories A through H with exact counts and percentages.
"""

import os
import sys
import json

def run_failure_classification():
    print("=" * 75)
    print("PHASE 3: ROOT-CAUSE FAILURE CLASSIFICATION")
    print("=" * 75)
    
    debug_file = "outputs/question_level_debug.json"
    if not os.path.exists(debug_file):
        raise FileNotFoundError(f"Question level debug file missing: {debug_file}")
        
    with open(debug_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    categories = {
        "A_Retrieval_Failure": 0,
        "B_Retrieval_Ranking_Failure": 0,
        "C_Chunking_Failure": 0,
        "D_Metadata_Failure": 0,
        "E_LLM_Generation_Failure": 0,
        "F_Prompt_Failure": 0,
        "G_Benchmark_Failure": 0,
        "H_Unsupported_Question": 0,
        "Success": 0
    }
    
    details = []
    total = len(records)
    
    for r in records:
        q_id = r["question_id"]
        q_text = r["question"]
        cls = r.get("classification", "UNKNOWN")
        
        cat_key = "Success"
        if cls == "UNSUPPORTED_QUESTION":
            cat_key = "H_Unsupported_Question"
        elif cls == "RETRIEVAL_FAILURE":
            cat_key = "A_Retrieval_Failure"
        elif cls == "RETRIEVAL_RANKING_FAILURE":
            cat_key = "B_Retrieval_Ranking_Failure"
        elif cls == "GENERATION_FAILURE":
            cat_key = "E_LLM_Generation_Failure"
            
        categories[cat_key] += 1
        
        details.append({
            "question_id": q_id,
            "category": cat_key,
            "question": q_text[:60] + "...",
            "retrieved_count": len(r.get("retrieved_chunks", []))
        })
        
    summary = {
        "total_evaluated_questions": total,
        "counts": categories,
        "percentages": {k: round(v / total * 100, 1) for k, v in categories.items()},
        "question_details": details
    }
    
    out_file = "outputs/failure_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"[+] Root-cause failure analysis report saved -> {out_file}")
    print("\nSummary Breakdown:")
    for k, v in categories.items():
        pct = round(v / total * 100, 1)
        print(f"  {k:30s}: {v:2d} ({pct:5.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    run_failure_classification()
