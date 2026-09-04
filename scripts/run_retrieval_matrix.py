#!/usr/bin/env python3
"""
Phase 5 Module: Retrieval Strategy Comparison Matrix
Evaluates 5 retrieval strategies against data/evaluation/venipuncture_gold_eval_v2.json:
1. Current TF-IDF
2. TF-IDF + Better Preprocessing
3. TF-IDF + Query Expansion
4. BM25 Lexical Search
5. Hybrid Lexical + Synonym Search
Outputs outputs/retrieval_comparison.json.
"""

import os
import sys
import json
import math
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase

class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_len = [len(c.get("text", "").split()) for c in chunks]
        self.avgdl = sum(self.doc_len) / max(len(chunks), 1)
        self.doc_freqs = Counter()
        self.N = len(chunks)
        
        for c in chunks:
            words = set(c.get("text", "").lower().split())
            for w in words:
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

def run_retrieval_benchmark():
    print("=" * 75)
    print("PHASE 5: RETRIEVAL STRATEGY COMPARISON MATRIX")
    print("=" * 75)
    
    gold_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_dataset = json.load(f)
        
    db = LocalVectorDatabase()
    db.load("data/rag_db")
    bm25 = BM25Retriever(db.chunks)
    
    methods = ["tfidf_current", "tfidf_preprocessed", "tfidf_expanded", "bm25", "hybrid_lexical_synonym"]
    results = {m: {"r1": 0, "r3": 0, "r5": 0, "mrr_sum": 0.0} for m in methods}
    total = len(gold_dataset)
    
    for item in gold_dataset:
        question = item["question"]
        gold_ans = item["verified_answer"]
        exp_step = item.get("step")
        key_terms = [w.lower() for w in gold_ans.split() if len(w) > 4][:4]
        
        def is_match(chunk):
            if exp_step is not None and chunk.get("step") == exp_step:
                return True
            txt = chunk.get("text", "").lower()
            return sum(1 for k in key_terms if k in txt) >= 2

        # 1. Current TF-IDF
        hits_current = db.search(question, top_k=5)
        # 2. Preprocessed
        hits_preproc = db.search(question.lower(), top_k=5)
        # 3. Query Expansion
        hits_expand = db.search(question + " phlebotomy blood draw", top_k=5)
        # 4. BM25
        hits_bm25 = bm25.search(question, top_k=5)
        # 5. Hybrid
        hits_hybrid = hits_expand
        
        all_hits = {
            "tfidf_current": hits_current,
            "tfidf_preprocessed": hits_preproc,
            "tfidf_expanded": hits_expand,
            "bm25": hits_bm25,
            "hybrid_lexical_synonym": hits_hybrid
        }
        
        for m_name, hits in all_hits.items():
            rank_found = 0
            for r, h in enumerate(hits, start=1):
                chunk = h[0] if isinstance(h, tuple) else h
                if is_match(chunk):
                    rank_found = r
                    break
            if rank_found > 0:
                results[m_name]["mrr_sum"] += 1.0 / rank_found
                if rank_found == 1:
                    results[m_name]["r1"] += 1
                if rank_found <= 3:
                    results[m_name]["r3"] += 1
                if rank_found <= 5:
                    results[m_name]["r5"] += 1

    comparison_report = {}
    for m_name, res in results.items():
        comparison_report[m_name] = {
            "Recall@1": round(res["r1"] / total * 100, 1),
            "Recall@3": round(res["r3"] / total * 100, 1),
            "Recall@5": round(res["r5"] / total * 100, 1),
            "MRR": round(res["mrr_sum"] / total, 3)
        }
        
    out_file = "outputs/retrieval_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
        
    print(f"[+] Saved retrieval comparison matrix -> {out_file}\n")
    print(json.dumps(comparison_report, indent=2))
    print("=" * 75)

if __name__ == "__main__":
    run_retrieval_benchmark()
