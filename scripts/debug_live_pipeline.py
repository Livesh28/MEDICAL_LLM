#!/usr/bin/env python3
"""
Live RAG Pipeline Debugger
Generates outputs/live_query_retrieval_debug.json and outputs/live_generation_debug.json
to trace query normalization, BM25 retrieval, prompt construction, model engine selection,
special token stripping, and metadata verification.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pipeline import MedicalRAGPipeline, sanitize_response_text
from rag.normalizer import normalize_query
from rag.retriever_v2 import tokenize_clean_text
from api.intent_router import classify_intent, format_deterministic_vr_response, INTENT_CLINICAL_QA, INTENT_OPEN_QUESTION

TEST_DEBUG_QUESTIONS = [
    ("What is hypertension?", 0, "General", None),
    ("What is diabetes?", 0, "General", None),
    ("What is venipuncture?", 0, "General", None),
    ("Why is the venipuncture site cleaned?", 5, "Clean Area", None),
    ("What is a tourniquet?", 2, "Apply Tourniquet", None),
    ("What is blood collection?", 12, "Blood Collection", None),
    ("What should I do at step 11?", 11, "Insert Tube", None)
]

def run_retrieval_and_generation_debug():
    print("[+] Initializing Live Pipeline Debugger...", flush=True)
    rag = MedicalRAGPipeline()
    
    vr_spec_path = "data/vr_knowledge/venipuncture_16_steps.json"
    vr_data = {}
    if os.path.exists(vr_spec_path):
        with open(vr_spec_path, "r", encoding="utf-8") as f:
            vr_data = {s["step"]: s for s in json.load(f).get("steps", [])}

    retrieval_debug_list = []
    generation_debug_list = []

    for idx, (question, step, step_name, mistake) in enumerate(TEST_DEBUG_QUESTIONS):
        print(f"\n[{idx+1}/{len(TEST_DEBUG_QUESTIONS)}] Debugging Query: '{question}'", flush=True)
        
        # 1. Trace Query Normalization & Tokenization
        norm_q = normalize_query(question)
        q_tokens = tokenize_clean_text(norm_q)
        intent = classify_intent(question)

        # 2. Trace Retrieval
        scored_hits = rag.hybrid_retriever.search(query=question, top_k=5, current_step=step)
        
        retrieved_chunks_debug = []
        top_score = scored_hits[0][1] if scored_hits else 0.0
        
        for chk_idx, (chunk, score) in enumerate(scored_hits):
            cid = chunk.get("chunk_id", chunk.get("id", f"chunk_{chk_idx}"))
            is_relevant = (score >= 1.0) and any(w in chunk.get("text", "").lower() for w in q_tokens)
            
            retrieved_chunks_debug.append({
                "rank": chk_idx + 1,
                "chunk_id": cid,
                "score": round(score, 4),
                "topic": chunk.get("topic", "General"),
                "source": chunk.get("source", "SRC_CLSI_01"),
                "text_snippet": chunk.get("text", "")[:180] + "...",
                "is_conceptually_relevant": is_relevant
            })

        retrieval_entry = {
            "query": question,
            "normalized_query": norm_q,
            "query_tokens": q_tokens,
            "intent": intent,
            "top_retrieved_score": round(top_score, 4),
            "top_5_retrieved_chunks": retrieved_chunks_debug
        }
        retrieval_debug_list.append(retrieval_entry)

        # 3. Trace Execution & Generation
        t0 = time.time()
        res = rag.answer_question(question, current_step=step, step_name=step_name, last_mistake=mistake)
        gen_time_ms = round((time.time() - t0) * 1000, 2)

        # Sanity check for token leakage
        leaked_tokens = [tok for tok in ["<|endoftext|>", "<|eos|>", "<|bos|>", "<|pad|>", "### Instruction"] if tok in res.get("answer", "")]

        gen_entry = {
            "query": question,
            "intent": intent,
            "engine": res.get("engine"),
            "grounded": res.get("grounded"),
            "confidence": res.get("confidence"),
            "raw_answer": res.get("answer"),
            "sanitized_answer": sanitize_response_text(res.get("answer", "")),
            "token_leakage_detected": leaked_tokens,
            "latency_ms": gen_time_ms,
            "first_source_chunk_id": res.get("sources", [{}])[0].get("chunk_id") if res.get("sources") else None
        }
        generation_debug_list.append(gen_entry)

    os.makedirs("outputs", exist_ok=True)
    
    retrieval_path = "outputs/live_query_retrieval_debug.json"
    with open(retrieval_path, "w", encoding="utf-8") as f:
        json.dump({"total_queries": len(TEST_DEBUG_QUESTIONS), "queries": retrieval_debug_list}, f, indent=2)

    gen_path = "outputs/live_generation_debug.json"
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump({"total_queries": len(TEST_DEBUG_QUESTIONS), "generations": generation_debug_list}, f, indent=2)

    print(f"\n[+] Live Pipeline Debug Complete.")
    print(f"    - Retrieval Debug Saved To: {retrieval_path}")
    print(f"    - Generation Debug Saved To: {gen_path}")

if __name__ == "__main__":
    run_retrieval_and_generation_debug()
