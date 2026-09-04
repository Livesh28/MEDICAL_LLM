#!/usr/bin/env python3
"""
Phase 14 Test Suite: Real Questions Audit
Executes 8 required Phase 14 test questions through POST /ask and verifies responses.
"""

import requests
import json

URL = "http://127.0.0.1:8000"

QUESTIONS = [
    {"id": "TEST 1", "question": "What is Step 0 in the venipuncture training workflow?", "current_step": 0},
    {"id": "TEST 2", "question": "What is Step 8?", "current_step": 8},
    {"id": "TEST 3", "question": "What is Step 12?", "current_step": 12},
    {"id": "TEST 4", "question": "What is the role of StepManager?"},
    {"id": "TEST 5", "question": "What is VeinTrigger used for?"},
    {"id": "TEST 6", "question": "What is the difference between VeinTrigger and BloodTrigger?"},
    {"id": "TEST 7", "question": "Why is hand hygiene important before venipuncture?"},
    {"id": "TEST 8", "question": "Explain the complete 16-step workflow."}
]

def run_phase14_tests():
    print("============================================================")
    print("PHASE 14: REAL QUESTIONS AUDIT")
    print("============================================================\n")

    results = []
    for item in QUESTIONS:
        q_id = item["id"]
        q_text = item["question"]
        payload = {"question": q_text, "model": "openbiollm", "current_step": item.get("current_step")}
        
        try:
            res = requests.post(f"{URL}/ask", json=payload, timeout=25).json()
            answer = res.get("answer", "")
            engine = res.get("engine", res.get("model"))
            sources = res.get("sources", [])

            record = {
                "id": q_id,
                "question": q_text,
                "engine": engine,
                "sources_count": len(sources),
                "answer": answer
            }
            results.append(record)

            print(f"[{q_id}] Engine: {engine:30s} | Sources: {len(sources)}")
            print(f"  Q: {q_text}")
            print(f"  A: {answer}\n")
        except Exception as e:
            print(f"[{q_id}] ERROR: {e}\n")

    out_file = "outputs/phase14_test_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Saved Phase 14 results to: {out_file}")

if __name__ == "__main__":
    run_phase14_tests()
