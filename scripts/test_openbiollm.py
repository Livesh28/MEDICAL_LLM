#!/usr/bin/env python3
"""
Test Script: OpenBioLLM Model Integration Test
Queries richardyoung/openbiollm:latest via Ollama on 5 benchmark clinical/VR questions.
Saves exact outputs to outputs/openbiollm_test.json.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.model_provider import OllamaModelProvider

TEST_QUESTIONS = [
    {"id": "q1", "question": "What is venipuncture?"},
    {"id": "q2", "question": "Why is the venipuncture site cleaned?"},
    {"id": "q3", "question": "What is a tourniquet?"},
    {"id": "q4", "question": "What is a blood collection tube?"},
    {"id": "q5", "question": "What should I do at step 11?"}
]


def run_openbiollm_test():
    print("=" * 70)
    print("OPENBIOLLM-8B INDEPENDENT MODEL TEST")
    print("Model: richardyoung/openbiollm:latest via Ollama (http://127.0.0.1:11434)")
    print("=" * 70)

    provider = OllamaModelProvider(model_name="richardyoung/openbiollm:latest")
    
    if not provider.is_available():
        print("[!] ERROR: richardyoung/openbiollm:latest is not available on local Ollama server!")
        sys.exit(1)

    print("[✓] Ollama model richardyoung/openbiollm:latest verified available.\n")

    results = []
    
    for item in TEST_QUESTIONS:
        q_id = item["id"]
        question = item["question"]
        print(f"[{q_id}] Question: {question}")
        
        prompt = (
            "You are an expert clinical phlebotomy instructor for a VR medical simulation.\n"
            "Answer the following question clearly, concisely, and medically accurately:\n\n"
            f"Question: {question}\n\n"
            "Medical Answer:"
        )
        
        t0 = time.time()
        try:
            answer = provider.generate(prompt=prompt, max_tokens=150, temperature=0.3)
            latency_ms = round((time.time() - t0) * 1000, 2)
            status = "success"
        except Exception as e:
            answer = f"[Error: {e}]"
            latency_ms = round((time.time() - t0) * 1000, 2)
            status = "error"
            
        print(f"     Latency: {latency_ms} ms")
        print(f"     Answer:  {answer[:120]}...\n")
        
        results.append({
            "id": q_id,
            "question": question,
            "answer": answer,
            "status": status,
            "latency_ms": latency_ms,
            "model": "richardyoung/openbiollm:latest",
            "provider": "ollama"
        })

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "openbiollm_test.json")
    
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "richardyoung/openbiollm:latest",
        "provider": "ollama",
        "total_questions": len(results),
        "results": results
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("=" * 70)
    print(f"[✓] OpenBioLLM test complete! Results saved to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_openbiollm_test()
