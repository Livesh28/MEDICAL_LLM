#!/usr/bin/env python3
"""
Phase 8 Module: Answer Grounding Checker
Post-generation QA guardrail that validates LLM output against retrieved evidence chunks
to catch unsupported patient claims or invented medical facts before returning to Unity.
"""

import re
from typing import List, Dict, Any, Tuple

UNSUPPORTED_PATTERNS = [
    r"\b120/80\b", r"\b130/85\b", r"\bblood pressure is\b", r"\bprescribed\b",
    r"\bpatient is \d+ years old\b", r"\bmedical history shows\b", r"\btaking medication\b"
]

SAFE_FALLBACK = "I don't have enough verified information in the current knowledge base to answer that reliably."

def validate_answer_grounding(
    generated_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    question: str
) -> Tuple[str, bool, str]:
    """
    Validates generated answer against evidence.
    Returns (final_answer, is_grounded, confidence_level).
    """
    gen_lower = generated_answer.lower().strip()
    
    # 1. Catch unsupported patient entity fabrications
    for pattern in UNSUPPORTED_PATTERNS:
        if re.search(pattern, gen_lower):
            print(f"[!] Grounding Checker Flagged Pattern: {pattern}")
            return SAFE_FALLBACK, False, "low"
            
    # 2. Catch empty or generic refusal responses
    if any(p in gen_lower for p in ["not provided", "not available", "don't have", "not mentioned", "unable to determine", "not specified"]):
        return generated_answer, True, "high"
        
    # 3. Keyword grounding overlap check against retrieved text
    combined_evidence = " ".join([c.get("text", "").lower() for c in retrieved_chunks])
    
    # Ignore system template / prompt meta words
    meta_stopwords = {
        "should", "first", "second", "third", "which", "where", "system", "based", "evidence",
        "provided", "according", "information", "description", "detail", "procedure", "trainee",
        "please", "context", "question", "answer", "simulation", "training", "vr_ground_truth"
    }
    
    gen_words = [re.sub(r"[^\w]", "", w) for w in gen_lower.split() if len(w) > 3 and re.sub(r"[^\w]", "", w) not in meta_stopwords]
    
    if gen_words:
        matches = sum(1 for w in gen_words if w in combined_evidence)
        match_ratio = matches / len(gen_words)
        
        # Check if any topic key concept from retrieved chunks is present in generated text
        has_topic_match = any(
            t_word in gen_lower
            for c in retrieved_chunks
            for t_word in re.findall(r"\b\w{4,}\b", c.get("topic", "") + " " + c.get("text", ""))
            if t_word.lower() not in meta_stopwords
        )

        if match_ratio < 0.05 and not has_topic_match and len(gen_words) >= 4:
            print(f"[!] Grounding Checker Flagged Low Evidence Overlap (Ratio: {match_ratio:.2f})")
            return SAFE_FALLBACK, False, "low"
            
    return generated_answer, True, "high"

if __name__ == "__main__":
    # Test grounding checker
    dummy_chunks = [{"text": "Hand hygiene must be performed before putting on gloves."}]
    test_gen_1 = "Clean non-sterile gloves should be worn after hand hygiene."
    test_gen_2 = "The patient's blood pressure is 120/80 mmHg."
    
    ans1, ok1, conf1 = validate_answer_grounding(test_gen_1, dummy_chunks, "When to wear gloves?")
    ans2, ok2, conf2 = validate_answer_grounding(test_gen_2, dummy_chunks, "What is blood pressure?")
    
    print(f"Test 1 Grounded: {ok1} -> {ans1[:50]}...")
    print(f"Test 2 Grounded: {ok2} -> {ans2[:50]}...")
