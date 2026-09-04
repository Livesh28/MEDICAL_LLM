#!/usr/bin/env python3
"""
Phase 5 Module: RAG V2 Retrieval Evaluation Comparison Matrix Script
Evaluates 4 retrieval variants against data/evaluation/venipuncture_gold_eval_v2.json:
1. BM25 Baseline
2. BM25 + Normalization
3. BM25 + Metadata Ranking
4. Hybrid Retrieval (RAG V2 Engine)
Outputs outputs/rag_v2_retrieval_comparison.json.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase
from rag.retriever_v2 import MetadataAwareHybridRetriever

def run_evaluation():
    print("=" * 75)
    print("PHASE 5: RAG V2 HYBRID RETRIEVAL COMPARISON BENCHMARK")
    print("=" * 75)
    
    gold_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_dataset = json.load(f)
        
    db = LocalVectorDatabase()
    db.load("data/rag_db")
    retriever = MetadataAwareHybridRetriever(db.chunks)
    
    variants = [
        "1_bm25_baseline",
        "2_bm25_normalized",
        "3_bm25_metadata",
        "4_hybrid_rag_v2"
    ]
    
    stats = {v: {"r1": 0, "r3": 0, "r5": 0, "mrr_sum": 0.0, "step_match": 0, "topic_match": 0} for v in variants}
    total = len(gold_dataset)
    
    for item in gold_dataset:
        question = item["question"]
        gold_ans = item["verified_answer"]
        exp_step = item.get("step")
        exp_topic = item.get("topic", "").lower()
        key_terms = [w.lower() for w in gold_ans.split() if len(w) > 4][:4]
        
        def evaluate_hits(hits):
            rank_found = 0
            for r, (chunk, score) in enumerate(hits, start=1):
                matches = sum(1 for k in key_terms if k in chunk.get("text", "").lower())
                is_match = (exp_step is not None and chunk.get("step") == exp_step) or (matches >= 2)
                if is_match and rank_found == 0:
                    rank_found = r
                    
            h_step_correct = hits[0][0].get("step") == exp_step if (exp_step is not None and len(hits)>0) else False
            h_topic_correct = hits[0][0].get("topic", "").lower() == exp_topic if len(hits)>0 else False
            return rank_found, h_step_correct, h_topic_correct

        # Variant 1: BM25 Baseline
        hits1 = retriever.search(question, top_k=5, use_normalization=False, use_metadata=False)
        r1, s1, t1 = evaluate_hits(hits1)
        
        # Variant 2: BM25 + Normalization
        hits2 = retriever.search(question, top_k=5, use_normalization=True, use_metadata=False)
        r2, s2, t2 = evaluate_hits(hits2)
        
        # Variant 3: BM25 + Metadata
        hits3 = retriever.search(question, top_k=5, current_step=exp_step, use_normalization=False, use_metadata=True)
        r3, s3, t3 = evaluate_hits(hits3)
        
        # Variant 4: Hybrid RAG V2 (BM25 + Normalization + Metadata)
        hits4 = retriever.search(question, top_k=5, current_step=exp_step, use_normalization=True, use_metadata=True)
        r4, s4, t4 = evaluate_hits(hits4)
        
        var_hits = {
            "1_bm25_baseline": (r1, s1, t1),
            "2_bm25_normalized": (r2, s2, t2),
            "3_bm25_metadata": (r3, s3, t3),
            "4_hybrid_rag_v2": (r4, s4, t4)
        }
        
        for v_name, (rank, s_ok, t_ok) in var_hits.items():
            if rank > 0:
                stats[v_name]["mrr_sum"] += 1.0 / rank
                if rank == 1:
                    stats[v_name]["r1"] += 1
                if rank <= 3:
                    stats[v_name]["r3"] += 1
                if rank <= 5:
                    stats[v_name]["r5"] += 1
            if s_ok:
                stats[v_name]["step_match"] += 1
            if t_ok:
                stats[v_name]["topic_match"] += 1

    comparison_report = {}
    for v_name, s_dict in stats.items():
        comparison_report[v_name] = {
            "Recall@1": round(s_dict["r1"] / total * 100, 1),
            "Recall@3": round(s_dict["r3"] / total * 100, 1),
            "Recall@5": round(s_dict["r5"] / total * 100, 1),
            "MRR": round(s_dict["mrr_sum"] / total, 3),
            "Step_Precision": round(s_dict["step_match"] / total * 100, 1),
            "Topic_Precision": round(s_dict["topic_match"] / total * 100, 1)
        }
        
    out_file = "outputs/rag_v2_retrieval_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
        
    print(f"[+] Saved RAG V2 retrieval comparison matrix -> {out_file}\n")
    print(json.dumps(comparison_report, indent=2))
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()
