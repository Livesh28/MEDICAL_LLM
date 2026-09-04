#!/usr/bin/env python3
"""
Phase 4 Module: 50-Query Intent Router Verification Script
Verifies intent classification across at least 50 test queries and generates outputs/intent_test_report.json.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.intent_router import (
    classify_intent,
    INTENT_NEXT_STEP,
    INTENT_REPEAT,
    INTENT_WHY_WRONG,
    INTENT_HELP,
    INTENT_VR_CONTEXT,
    INTENT_CLINICAL_QA,
    INTENT_OPEN_QUESTION,
    INTENT_UNSUPPORTED
)

# 50 Test Queries covering all supported intents
TEST_QUERIES = [
    # NEXT_STEP (7 queries)
    ("What should I do next?", INTENT_NEXT_STEP),
    ("What do I do next?", INTENT_NEXT_STEP),
    ("What is the next step?", INTENT_NEXT_STEP),
    ("What now?", INTENT_NEXT_STEP),
    ("Where to start?", INTENT_NEXT_STEP),
    ("What's next?", INTENT_NEXT_STEP),
    ("Tell me the next step", INTENT_NEXT_STEP),

    # REPEAT (6 queries)
    ("Repeat that", INTENT_REPEAT),
    ("Pardon", INTENT_REPEAT),
    ("Say again", INTENT_REPEAT),
    ("What did you say?", INTENT_REPEAT),
    ("Repeat the instruction", INTENT_REPEAT),
    ("Say step again", INTENT_REPEAT),

    # WHY_WRONG (6 queries)
    ("Why was that wrong?", INTENT_WHY_WRONG),
    ("Why did I get an error?", INTENT_WHY_WRONG),
    ("What mistake did I make?", INTENT_WHY_WRONG),
    ("Why failed?", INTENT_WHY_WRONG),
    ("What went wrong?", INTENT_WHY_WRONG),
    ("Why did that fail?", INTENT_WHY_WRONG),

    # HELP (5 queries)
    ("Help me", INTENT_HELP),
    ("I am stuck", INTENT_HELP),
    ("I'm stuck", INTENT_HELP),
    ("Where is the annotator guidance?", INTENT_HELP),
    ("Show me where to go", INTENT_HELP),

    # VR_CONTEXT (5 queries)
    ("Which object should I use?", INTENT_VR_CONTEXT),
    ("What object do I grab?", INTENT_VR_CONTEXT),
    ("Where should I put the tube?", INTENT_VR_CONTEXT),
    ("Where do I place the tourniquet?", INTENT_VR_CONTEXT),
    ("Where does this go?", INTENT_VR_CONTEXT),

    # UNSUPPORTED (8 queries)
    ("What is the patient's blood pressure?", INTENT_UNSUPPORTED),
    ("What medication does the patient take?", INTENT_UNSUPPORTED),
    ("What is the patient's medical history?", INTENT_UNSUPPORTED),
    ("How old is the patient?", INTENT_UNSUPPORTED),
    ("What are the lab results?", INTENT_UNSUPPORTED),
    ("Can we skip step 5?", INTENT_UNSUPPORTED),
    ("Do a capillary stick instead", INTENT_UNSUPPORTED),
    ("What is the patient name?", INTENT_UNSUPPORTED),

    # OPEN_QUESTION (5 queries)
    ("Tell me about venipuncture", INTENT_OPEN_QUESTION),
    ("Explain phlebotomy safety", INTENT_OPEN_QUESTION),
    ("Overview of CLSI standards", INTENT_OPEN_QUESTION),
    ("Describe the order of draw", INTENT_OPEN_QUESTION),
    ("Tell me about tourniquet application", INTENT_OPEN_QUESTION),

    # CLINICAL_QA (8 queries)
    ("Why do we clean the site with alcohol?", INTENT_CLINICAL_QA),
    ("What is the maximum time a tourniquet can remain on?", INTENT_CLINICAL_QA),
    ("What angle should the needle be inserted?", INTENT_CLINICAL_QA),
    ("Why must alcohol dry for 30 seconds?", INTENT_CLINICAL_QA),
    ("What tube is drawn first for blood culture?", INTENT_CLINICAL_QA),
    ("What causes hemolysis during blood draw?", INTENT_CLINICAL_QA),
    ("What is the function of sodium citrate in blue tubes?", INTENT_CLINICAL_QA),
    ("Why must tubes be inverted gently?", INTENT_CLINICAL_QA),
]

def run_intent_router_evaluation():
    print(f"[+] Starting Intent Router Evaluation ({len(TEST_QUERIES)} Queries)...")
    results = []
    correct_count = 0
    total_latency_ms = 0.0

    for query_text, expected in TEST_QUERIES:
        t0 = time.time()
        predicted = classify_intent(query_text)
        latency = (time.time() - t0) * 1000
        total_latency_ms += latency
        
        is_correct = (predicted == expected)
        if is_correct:
            correct_count += 1
            
        results.append({
            "text": query_text,
            "expected_intent": expected,
            "predicted_intent": predicted,
            "correct": is_correct,
            "latency_ms": round(latency, 3)
        })

    accuracy = correct_count / len(TEST_QUERIES)
    avg_latency = total_latency_ms / len(TEST_QUERIES)

    output_report = {
        "total_queries": len(TEST_QUERIES),
        "correct_predictions": correct_count,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(avg_latency, 3),
        "test_results": results
    }

    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/intent_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    print(f"[+] Intent Router Evaluation Complete.")
    print(f"    - Accuracy: {accuracy * 100:.2f}% ({correct_count}/{len(TEST_QUERIES)})")
    print(f"    - Avg Latency: {avg_latency:.3f} ms")
    print(f"    - Report Saved To: {report_path}")
    return output_report

if __name__ == "__main__":
    run_intent_router_evaluation()
