#!/usr/bin/env python3
"""
Multi-Model Live Demo Verification Script
Queries running FastAPI Server at http://127.0.0.1:8000 across:
1. /health system telemetry
2. /ask with OpenBioLLM-8B (Production Candidate)
3. /ask with Llama 3.2 3B (Benchmark Fallback)
4. /ask with MedicalTransformerLM 110M (PyTorch Research Offline)
5. /ask with AUTO Smart Fallback Mode
6. /ask VR Deterministic Next Step Query
"""

import json
import requests
import time

SERVER_URL = "http://127.0.0.1:8000"


def run_live_demo():
    print("=" * 75)
    print("      MULTI-MODEL MEDICAL LLM + VR ASSISTANT — LIVE DEMO")
    print("=" * 75)
    print(f"Target Server URL: {SERVER_URL}\n")

    # 1. Health Check
    try:
        res = requests.get(f"{SERVER_URL}/health", timeout=5)
        print("[1/6] GET /health Telemetry Status:")
        print(json.dumps(res.json(), indent=2))
        print("-" * 75)
    except Exception as e:
        print(f"[!] Health check failed (Is server running on localhost:8000?): {e}")
        return

    question = "Why is the venipuncture site cleaned with 70% isopropyl alcohol?"

    # 2. OpenBioLLM-8B Query
    payload_ob = {"question": question, "model": "openbiollm"}
    t0 = time.time()
    res_ob = requests.post(f"{SERVER_URL}/ask", json=payload_ob, timeout=60).json()
    dt_ob = round((time.time() - t0) * 1000, 2)
    print("[2/6] POST /ask (Model: OpenBioLLM-8B Production Candidate):")
    print(f"  - Provider: {res_ob.get('provider')} | Model: {res_ob.get('model')}")
    print(f"  - Grounded: {res_ob.get('grounded')} | Confidence: {res_ob.get('confidence')}")
    print(f"  - Answer:   \"{res_ob.get('answer')}\"")
    print(f"  - Latency:  {dt_ob} ms")
    print("-" * 75)

    # 3. Llama 3.2 3B Query
    payload_l3 = {"question": question, "model": "llama32"}
    t0 = time.time()
    res_l3 = requests.post(f"{SERVER_URL}/ask", json=payload_l3, timeout=30).json()
    dt_l3 = round((time.time() - t0) * 1000, 2)
    print("[3/6] POST /ask (Model: Llama 3.2 3B Benchmark Fallback):")
    print(f"  - Provider: {res_l3.get('provider')} | Model: {res_l3.get('model')}")
    print(f"  - Grounded: {res_l3.get('grounded')} | Confidence: {res_l3.get('confidence')}")
    print(f"  - Answer:   \"{res_l3.get('answer')}\"")
    print(f"  - Latency:  {dt_l3} ms")
    print("-" * 75)

    # 4. MedicalTransformerLM 110M Query
    payload_pt = {"question": question, "model": "medical_transformer_110m"}
    t0 = time.time()
    res_pt = requests.post(f"{SERVER_URL}/ask", json=payload_pt, timeout=30).json()
    dt_pt = round((time.time() - t0) * 1000, 2)
    print("[4/6] POST /ask (Model: MedicalTransformerLM 110M PyTorch Offline):")
    print(f"  - Provider: {res_pt.get('provider')} | Model: {res_pt.get('model')} | Parameters: {res_pt.get('parameters')}")
    print(f"  - Grounded: {res_pt.get('grounded')} | Confidence: {res_pt.get('confidence')}")
    print(f"  - Answer:   \"{res_pt.get('answer')}\"")
    print(f"  - Latency:  {dt_pt} ms")
    print("-" * 75)

    # 5. AUTO Mode Query
    payload_auto = {"question": question, "model": "auto"}
    t0 = time.time()
    res_auto = requests.post(f"{SERVER_URL}/ask", json=payload_auto, timeout=60).json()
    dt_auto = round((time.time() - t0) * 1000, 2)
    print("[5/6] POST /ask (Model: AUTO Smart Fallback):")
    print(f"  - Resolved Provider: {res_auto.get('provider')} | Resolved Model: {res_auto.get('model')}")
    print(f"  - Answer:            \"{res_auto.get('answer')}\"")
    print(f"  - Latency:           {dt_auto} ms")
    print("-" * 75)

    # 6. VR Deterministic Query
    payload_vr = {"question": "What should I do next?", "current_step": 11, "step_name": "Insert Tube"}
    t0 = time.time()
    res_vr = requests.post(f"{SERVER_URL}/ask", json=payload_vr, timeout=10).json()
    dt_vr = round((time.time() - t0) * 1000, 2)
    print("[6/6] POST /ask (VR Deterministic Step Query):")
    print(f"  - Detected Intent: {res_vr.get('intent')}")
    print(f"  - VR Engine:       {res_vr.get('engine')}")
    print(f"  - Answer:          \"{res_vr.get('answer')}\"")
    print(f"  - Latency:         {dt_vr} ms")
    print("=" * 75)
    print("                 LIVE DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_live_demo()
