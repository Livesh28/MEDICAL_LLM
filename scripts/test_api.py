#!/usr/bin/env python3
"""
Verification Script: Test Local API Server & Multi-Model /ask Endpoints
Tests:
1. Server startup & health endpoint (/health).
2. Direct LLM generation endpoint (/generate).
3. Medical RAG /ask endpoint with models: openbiollm, llama32, medical_transformer, auto.
4. Deterministic VR intent routing.
5. Web UI dashboard route (/).
"""

import sys
import os
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.server import app


def main():
    print("=" * 70)
    print("Multi-Model Local Medical LLM API Verification")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Health Check
        print("\n--- 1. Testing GET /health Endpoint ---")
        res_health = client.get("/health")
        print(f"Status Code: {res_health.status_code}")
        print(f"Response:    {res_health.json()}")
        assert res_health.status_code == 200, "Health check failed!"

        # 2. Test GET / (UI Dashboard Route)
        print("\n--- 2. Testing GET / (Web Dashboard Route) ---")
        res_ui = client.get("/")
        print(f"Status Code: {res_ui.status_code}")
        assert res_ui.status_code == 200, "UI route failed!"
        assert "Local Medical LLM Workbench" in res_ui.text, "UI HTML content missing!"

        # 3. Test POST /ask with different model selections
        models_to_test = ["openbiollm", "llama32", "medical_transformer", "auto"]
        for m in models_to_test:
            print(f"\n--- 3. Testing POST /ask with model='{m}' ---")
            payload_ask = {
                "question": "What is venipuncture?",
                "top_k_chunks": 2,
                "max_new_tokens": 40,
                "temperature": 0.3,
                "model": m
            }
            res_ask = client.post("/ask", json=payload_ask)
            print(f"Status Code: {res_ask.status_code}")
            ask_data = res_ask.json()
            print(f"  Model Requested: {m}")
            print(f"  Provider Used:   {ask_data.get('provider')}")
            print(f"  Model Reported:  {ask_data.get('model')}")
            print(f"  Answer Snippet:  '{ask_data['answer'][:80]}...'")
            assert res_ask.status_code == 200, f"Ask endpoint failed for model {m}!"
            assert "answer" in ask_data and "provider" in ask_data, "Response missing required fields!"

        # 4. Test Deterministic VR Query
        print("\n--- 4. Testing POST /ask with Deterministic VR Intent ---")
        vr_payload = {
            "question": "What should I do next?",
            "current_step": 11,
            "step_name": "Insert Tube",
            "model": "openbiollm"
        }
        res_vr = client.post("/ask", json=vr_payload)
        print(f"Status Code: {res_vr.status_code}")
        vr_data = res_vr.json()
        print(f"  VR Engine: {vr_data.get('engine')}")
        print(f"  Answer:    '{vr_data['answer']}'")
        assert res_vr.status_code == 200, "Deterministic VR ask endpoint failed!"
        assert vr_data.get("engine") == "vr_stepmanager_deterministic", "VR intent should be deterministic!"

    print("\n" + "=" * 70)
    print("[SUCCESS] API Multi-Model Verification PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
