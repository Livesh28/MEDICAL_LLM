#!/usr/bin/env python3
"""
Phase 9 Module: Controlled 4-Condition Model Experiment Script
Evaluates Llama 3.2 3B under 4 controlled conditions:
- Condition A: Llama 3.2 3B Standalone
- Condition B: Llama 3.2 3B + Current TF-IDF RAG
- Condition C: Llama 3.2 3B + Improved BM25 Retrieval
- Condition D: Llama 3.2 3B + Improved BM25 Retrieval + Grounded Prompt v2
Outputs outputs/controlled_model_comparison.json.
"""

import os
import sys
import json
import requests
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

    def search(self, query: str, top_k: int = 3):
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

def score_answer(generated: str, gold: str, is_unsupported: bool = False) -> str:
    gen_lower = generated.lower().strip()
    if is_unsupported:
        if "not provided" in gen_lower or "not available" in gen_lower or "cannot" in gen_lower or "don't have" in gen_lower:
            return "Correct Refusal"
        return "Hallucinated"
        
    key_terms = [w.lower() for w in gold.split() if len(w) > 4][:5]
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

def run_controlled_experiment():
    print("=" * 75)
    print("PHASE 9: CONTROLLED 4-CONDITION MODEL EXPERIMENT")
    print("=" * 75)
    
    gold_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_dataset = json.load(f)
        
    db = LocalVectorDatabase()
    db.load("data/rag_db")
    bm25 = BM25Searcher(db.chunks)
    
    conditions = ["cond_A_standalone", "cond_B_tfidf_rag", "cond_C_bm25_rag", "cond_D_bm25_grounded_prompt"]
    stats = {c: {"Correct": 0, "Correct Refusal": 0, "Partially Correct": 0, "Incorrect": 0, "Hallucinated": 0} for c in conditions}
    
    for idx, item in enumerate(gold_dataset, start=1):
        q = item["question"]
        gold = item["verified_answer"]
        is_unsup = item.get("topic") == "unsupported_safeguard"
        
        # Condition A: Standalone
        p_A = f"Instruction: {q}\nMedical Answer:"
        ans_A = query_ollama(p_A)
        stats["cond_A_standalone"][score_answer(ans_A, gold, is_unsup)] += 1
        
        # Condition B: TF-IDF RAG
        hits_b = db.search(q, top_k=2)
        ctx_b = "\n".join([h[0]["text"][:200] for h in hits_b]) if hits_b else ""
        p_B = f"Context:\n{ctx_b}\n\nQuestion: {q}\nMedical Answer:"
        ans_B = query_ollama(p_B)
        stats["cond_B_tfidf_rag"][score_answer(ans_B, gold, is_unsup)] += 1
        
        # Condition C: BM25 RAG
        hits_c = bm25.search(q, top_k=2)
        ctx_c = "\n".join([h[0]["text"][:200] for h in hits_c]) if hits_c else ""
        p_C = f"Context:\n{ctx_c}\n\nQuestion: {q}\nMedical Answer:"
        ans_C = query_ollama(p_C)
        stats["cond_C_bm25_rag"][score_answer(ans_C, gold, is_unsup)] += 1
        
        # Condition D: BM25 RAG + Grounded Prompt v2
        p_D = (
            "You are an expert clinical phlebotomy instructor for a VR venipuncture simulator. "
            "Answer the question accurately using ONLY the clinical evidence below. "
            "If asked for patient-specific details (blood pressure, medications), state clearly that the information is not provided.\n\n"
            f"Clinical Evidence:\n{ctx_c}\n\n"
            f"Question: {q}\nMedical Answer:"
        )
        ans_D = query_ollama(p_D)
        stats["cond_D_bm25_grounded_prompt"][score_answer(ans_D, gold, is_unsup)] += 1
        
        print(f"[{idx:2d}/25] Tested across Conditions A-D...")

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
            "breakdown": m_dict
        }
        
    out_file = "outputs/controlled_model_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[+] Saved controlled model comparison -> {out_file}")
    print(json.dumps(report["conditions"], indent=2))
    print("=" * 75)

if __name__ == "__main__":
    run_controlled_experiment()
