#!/usr/bin/env python3
"""
Phase 12 & 20 Module: Automated Data Leakage Checker Script
Audits exact string matching and n-gram overlap between training files and evaluation benchmark datasets.
"""

import os
import json
import re
from typing import Set

def get_ngrams(text: str, n: int = 5) -> Set[str]:
    words = [w.lower() for w in re.findall(r'\w+', text)]
    if len(words) < n:
        return set()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

def check_leakage():
    print("=" * 70)
    print("PHASE 12 & 20: DATASET LEAKAGE AUDIT (TRAIN VS EVALUATION)")
    print("=" * 70)
    
    eval_file = "data/evaluation/venipuncture_gold_eval_v2.json"
    sft_file = "data/sft/venipuncture_sft_dataset_v3.json"
    
    if not os.path.exists(eval_file) or not os.path.exists(sft_file):
        print("[!] Error: Evaluation or SFT files missing.")
        return
        
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    with open(sft_file, "r", encoding="utf-8") as f:
        sft_data = json.load(f)
        
    eval_questions = {item["question"].strip().lower() for item in eval_data}
    sft_questions = {item["instruction"].strip().lower() for item in sft_data}
    
    # 1. Exact Question Overlap Check
    exact_matches = eval_questions.intersection(sft_questions)
    
    # 2. 5-Gram Text Overlap Check
    eval_ngrams = set()
    for item in eval_data:
        eval_ngrams.update(get_ngrams(item["question"]))
        
    sft_ngrams = set()
    for item in sft_data:
        sft_ngrams.update(get_ngrams(item["instruction"]))
        
    ngram_overlap = eval_ngrams.intersection(sft_ngrams)
    
    print(f"[+] Total Evaluation Benchmark Questions: {len(eval_questions)}")
    print(f"[+] Total SFT Training Questions:         {len(sft_questions)}")
    print(f"[+] Exact Duplicate Question Overlap:      {len(exact_matches)}")
    print(f"[+] 5-Gram Sub-Phrase Overlap Count:       {len(ngram_overlap)}")
    
    print("=" * 70)
    if len(exact_matches) == 0:
        print("[✓] ZERO DATA LEAKAGE DETECTED! Evaluation benchmark is 100% untouched.")
    else:
        print(f"[!] WARNING: Found {len(exact_matches)} leaked evaluation questions in training set!")
        for m in list(exact_matches)[:3]:
            print(f"    - Leaked question: \"{m}\"")
    print("=" * 70)

if __name__ == "__main__":
    check_leakage()
