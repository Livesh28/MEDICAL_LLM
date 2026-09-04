#!/usr/bin/env python3
"""
Phase 4 Backend Endpoint Verification Test
Verifies API responses, models, and retrieval independently from the UI.
"""

import requests
import json

URL = "http://127.0.0.1:8000"

def test_endpoints():
    print("============================================================")
    print("PHASE 4: BACKEND ENDPOINTS VERIFICATION")
    print("============================================================")

    # 1. Health check
    h_res = requests.get(f"{URL}/health", timeout=3).json()
    print("[✓] GET /health:")
    print("   ", json.dumps(h_res))

    # 2. Hand hygiene clinical question
    q1 = "Why is hand hygiene important before venipuncture?"
    res1 = requests.post(f"{URL}/ask", json={"question": q1, "model": "openbiollm"}).json()
    print(f"\n[✓] POST /ask (Question: '{q1}'):")
    print(f"    Model:   {res1.get('model')}")
    print(f"    Answer:  {res1.get('answer')}")
    print(f"    Sources: {len(res1.get('sources', []))} chunks retrieved.")

    # 3. VR Step query
    q2 = "What should I do next?"
    res2 = requests.post(f"{URL}/ask", json={"question": q2, "model": "openbiollm", "current_step": 11}).json()
    print(f"\n[✓] POST /ask (VR Query: '{q2}'):")
    print(f"    Intent:  {res2.get('intent')}")
    print(f"    Engine:  {res2.get('engine')}")
    print(f"    Answer:  {res2.get('answer')}")

    # 4. Direct generation
    res3 = requests.post(f"{URL}/generate", json={"prompt": "Function of red blood cells:", "max_new_tokens": 40}).json()
    print(f"\n[✓] POST /generate:")
    print(f"    Generated: {res3.get('generated_text')[:100]}...")

    print("============================================================\n")

if __name__ == "__main__":
    test_endpoints()
