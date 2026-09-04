#!/usr/bin/env python3
"""
Phase 14 Module: Vector Database Retrieval Evaluation Script
Evaluates Recall@1, Recall@3, Recall@5, MRR, topic precision, and step precision on Gold Benchmark v2.
Outputs results to outputs/retrieval_eval_report.json.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase

def evaluate_retrieval(
    eval_file: str = "data/evaluation/venipuncture_gold_eval_v2.json",
    db_dir: str = "data/rag_db"
):
    print("=" * 75)
    print("PHASE 14: RAG VECTOR RETRIEVAL EVALUATION")
    print("=" * 75)
    
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation benchmark file not found at {eval_file}")
        
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_items = json.load(f)
        
    db = LocalVectorDatabase()
    db.load(db_dir)
    print(f"[+] Loaded Vector Database from {db_dir} with {len(db.chunks):,} total chunks.")
    
    r1_count = 0
    r3_count = 0
    r5_count = 0
    mrr_sum = 0.0
    
    step_correct_count = 0
    topic_correct_count = 0
    
    query_records = []
    
    for idx, item in enumerate(eval_items, start=1):
        question = item["question"]
        target_step = item.get("step")
        target_topic = item.get("topic", "").lower()
        verified_ans = item["verified_answer"].lower()
        
        # Search top 5 chunks
        hits = db.search(query=question, top_k=5)
        
        found_at_rank = 0
        hit_topics = []
        hit_steps = []
        
        for rank, item_hit in enumerate(hits, start=1):
            if isinstance(item_hit, tuple):
                chunk, score = item_hit
            else:
                chunk = item_hit
                
            text_lower = chunk.get("text", "").lower()
            h_step = chunk.get("step")
            h_topic = chunk.get("topic", "").lower()
            
            hit_topics.append(h_topic)
            if h_step is not None:
                hit_steps.append(h_step)
                
            # Check relevance by keyword matching against verified answer
            key_terms = [w.lower() for w in verified_ans.split() if len(w) > 4][:5]
            matches = sum(1 for k in key_terms if k in text_lower)
            
            if matches >= 2 and found_at_rank == 0:
                found_at_rank = rank
                
        if found_at_rank == 1:
            r1_count += 1
            r3_count += 1
            r5_count += 1
            mrr_sum += 1.0
        elif 1 < found_at_rank <= 3:
            r3_count += 1
            r5_count += 1
            mrr_sum += (1.0 / found_at_rank)
        elif 3 < found_at_rank <= 5:
            r5_count += 1
            mrr_sum += (1.0 / found_at_rank)
            
        if target_step in hit_steps[:3]:
            step_correct_count += 1
        if any(target_topic in t for t in hit_topics[:3]):
            topic_correct_count += 1
            
        query_records.append({
            "id": item["id"],
            "question": question,
            "found_at_rank": found_at_rank,
            "top_hit_text": (hits[0][0]["text"][:120] if isinstance(hits[0], tuple) else hits[0]["text"][:120]) if hits else ""
        })
        
    n = len(eval_items)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_queries": n,
        "metrics": {
            "recall_at_1_percent": round(r1_count / n * 100, 1),
            "recall_at_3_percent": round(r3_count / n * 100, 1),
            "recall_at_5_percent": round(r5_count / n * 100, 1),
            "mrr": round(mrr_sum / n, 3),
            "step_retrieval_accuracy_percent": round(step_correct_count / n * 100, 1),
            "topic_retrieval_accuracy_percent": round(topic_correct_count / n * 100, 1)
        },
        "query_details": query_records
    }
    
    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/retrieval_eval_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("=" * 75)
    print("RETRIEVAL EVALUATION SUMMARY REPORT")
    print("=" * 75)
    print(f" Recall @ 1: {report['metrics']['recall_at_1_percent']}%")
    print(f" Recall @ 3: {report['metrics']['recall_at_3_percent']}%")
    print(f" Recall @ 5: {report['metrics']['recall_at_5_percent']}%")
    print(f" MRR:        {report['metrics']['mrr']}")
    print(f" Step Precision:  {report['metrics']['step_retrieval_accuracy_percent']}%")
    print(f" Topic Precision: {report['metrics']['topic_retrieval_accuracy_percent']}%")
    print(f"\n[+] Saved retrieval report -> {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    evaluate_retrieval()
