#!/usr/bin/env python3
"""
Phase 2 Module: RAG V2 Retrieval Debugging Trace Generator
Evaluates data/evaluation/venipuncture_gold_eval_v2.json against Okapi BM25 search
and outputs outputs/rag_v2_question_debug.json for question-by-question manual inspection.
"""

import os
import sys
import json
import math
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase

class BM25Searcher:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_len = [len(c.get("text", "").split()) for c in chunks]
        self.avgdl = sum(self.doc_len) / max(len(chunks), 1)
        self.doc_freqs = Counter()
        self.N = len(chunks)
        for c in chunks:
            for w in set(c.get("text", "").lower().split()):
                self.doc_freqs[w] += 1

    def search(self, query: str, top_k: int = 5):
        q_words = query.lower().split()
        scores = []
        for idx, chunk in enumerate(self.chunks):
            score = 0.0
            doc_words = chunk.get("text", "").lower().split()
            word_counts = Counter(doc_words)
            for qw in q_words:
                if qw in word_counts:
                    df = self.doc_freqs.get(qw, 0)
                    idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                    tf = word_counts[qw]
                    len_norm = 1.0 - self.b + self.b * (self.doc_len[idx] / self.avgdl)
                    score += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
            scores.append((chunk, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def run_debug():
    print("=" * 75)
    print("PHASE 2: RAG V2 RETRIEVAL DEBUGGING TRACE GENERATOR")
    print("=" * 75)
    
    gold_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_dataset = json.load(f)
        
    db = LocalVectorDatabase()
    db.load("data/rag_db")
    bm25 = BM25Searcher(db.chunks)
    
    debug_records = []
    
    for item in gold_dataset:
        q_id = item["id"]
        q_text = item["question"]
        gold_ans = item["verified_answer"]
        exp_src = item.get("source_id", "SRC_CLSI_01")
        exp_topic = item.get("topic", "")
        exp_step = item.get("step")
        
        key_terms = [w.lower() for w in gold_ans.split() if len(w) > 4][:4]
        hits = bm25.search(q_text, top_k=5)
        
        retrieved_list = []
        correct_retrieved = False
        correct_rank = None
        
        for rank, (chunk, score) in enumerate(hits, start=1):
            c_text = chunk.get("text", "")
            matches = sum(1 for k in key_terms if k in c_text.lower())
            is_match = (exp_step is not None and chunk.get("step") == exp_step) or (matches >= 2)
            
            if is_match and not correct_retrieved:
                correct_retrieved = True
                correct_rank = rank
                
            retrieved_list.append({
                "rank": rank,
                "chunk_id": chunk.get("chunk_id", f"c_{rank}"),
                "score": round(score, 4),
                "topic": chunk.get("topic", "N/A"),
                "step": chunk.get("step"),
                "source_id": chunk.get("source", "N/A"),
                "text": c_text[:120] + "..."
            })
            
        debug_records.append({
            "question_id": q_id,
            "question": q_text,
            "expected_source": exp_src,
            "expected_topic": exp_topic,
            "expected_step": exp_step,
            "retrieved": retrieved_list,
            "correct_chunk_retrieved": correct_retrieved,
            "correct_rank": correct_rank
        })
        
    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/rag_v2_question_debug.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(debug_records, f, indent=2)
        
    print(f"[+] Saved {len(debug_records)} debug records -> {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    run_debug()
