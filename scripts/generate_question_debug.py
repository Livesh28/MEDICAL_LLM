#!/usr/bin/env python3
"""
Phase 2 Module: Question-Level Debug Record Generator
Generates outputs/question_level_debug.json tracing retrieval, generation, and root-cause failure classification
for every question in data/evaluation/venipuncture_gold_eval_v2.json.
"""

import os
import sys
import json
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def generate_debug_records():
    print("=" * 75)
    print("PHASE 2: QUESTION-LEVEL DEBUG RECORD GENERATION")
    print("=" * 75)
    
    gold_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    if not os.path.exists(gold_file):
        raise FileNotFoundError(f"Gold benchmark file missing: {gold_file}")
        
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_dataset = json.load(f)
        
    db = LocalVectorDatabase()
    db.load("data/rag_db")
    
    debug_records = []
    
    for item in gold_dataset:
        q_id = item["id"]
        question = item["question"]
        gold_ans = item["verified_answer"]
        exp_src = item.get("source", "SRC_CLSI_01")
        exp_topic = item.get("topic", "")
        exp_step = item.get("step")
        
        # 1. RAG Vector Search
        hits = db.search(query=question, top_k=3)
        retrieved_chunks = []
        retrieval_correct = False
        
        for rank, hit in enumerate(hits, start=1):
            chunk = hit[0] if isinstance(hit, tuple) else hit
            score = round(hit[1], 4) if isinstance(hit, tuple) else round(chunk.get("score", 0.0), 4)
            
            retrieved_chunks.append({
                "rank": rank,
                "chunk_id": chunk.get("chunk_id", f"c_{rank}"),
                "score": score,
                "text": chunk.get("text", "")[:120] + "...",
                "source_id": chunk.get("source", "N/A"),
                "topic": chunk.get("topic", "N/A"),
                "step": chunk.get("step")
            })
            
            # Check if retrieved chunk matches expected step or key terms
            if exp_step is not None and chunk.get("step") == exp_step:
                retrieval_correct = True
            key_terms = [w.lower() for w in gold_ans.split() if len(w) > 4][:4]
            if sum(1 for k in key_terms if k in chunk.get("text", "").lower()) >= 2:
                retrieval_correct = True
                
        # 2. LLM Generation
        snippets = [c["text"] for c in retrieved_chunks]
        context_str = "\n".join(snippets)
        prompt = (
            "You are an expert clinical phlebotomy instructor for a VR venipuncture simulator. "
            "Answer the question accurately using the clinical context below:\n\n"
            f"Clinical Context:\n{context_str}\n\n"
            f"Question: {question}\n\nMedical Answer:"
        )
        model_ans = query_ollama(prompt, model_name="llama3.2:3b", max_tokens=100)
        
        # 3. Assess Generation Correctness & Classify Failure
        key_terms_ans = [w.lower() for w in gold_ans.split() if len(w) > 4][:5]
        matches = sum(1 for k in key_terms_ans if k in model_ans.lower())
        
        generation_correct = (matches >= 2) or ("not available" in model_ans.lower() or "not provided" in model_ans.lower())
        
        if exp_topic == "unsupported_safeguard":
            classification = "UNSUPPORTED_QUESTION"
        elif not retrieval_correct and not generation_correct:
            classification = "RETRIEVAL_FAILURE"
        elif retrieval_correct and not generation_correct:
            classification = "GENERATION_FAILURE"
        elif not retrieval_correct and generation_correct:
            classification = "RETRIEVAL_RANKING_FAILURE"
        else:
            classification = "SUCCESS"
            
        debug_records.append({
            "question_id": q_id,
            "question": question,
            "gold_answer": gold_ans,
            "retrieved_chunks": retrieved_chunks,
            "expected_source": exp_src,
            "expected_topic": exp_topic,
            "expected_step": exp_step,
            "retrieval_correct": retrieval_correct,
            "generation_correct": generation_correct,
            "model_answer": model_ans,
            "classification": classification
        })
        
    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/question_level_debug.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(debug_records, f, indent=2)
        
    print(f"[+] Saved {len(debug_records)} question-level debug records -> {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    generate_debug_records()
