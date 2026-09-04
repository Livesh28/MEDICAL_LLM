#!/usr/bin/env python3
"""
Comprehensive System Verification Script
Audits all 10 core components of the Medical LLM & RAG System.
"""

import os
import sys
import json
import time
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.model_provider import ModelRouter, OllamaModelProvider, MedicalTransformerProvider
from rag.pipeline import MedicalRAGPipeline
from api.intent_router import classify_intent, format_deterministic_vr_response
from scripts.validate_grounding import validate_answer_grounding


def run_comprehensive_audit():
    print("============================================================")
    print("       MEDICAL LLM & RAG SYSTEM — COMPLETE AUDIT")
    print("============================================================\n")

    results = {}

    # Step 1: Check 110M Preserved Model Checkpoints & Hashes
    print("[1/10] Auditing Preserved 110M MedicalTransformerLM...")
    try:
        inv_path = "outputs/checkpoint_inventory.json"
        if os.path.exists(inv_path):
            with open(inv_path, "r") as f:
                inv = json.load(f)
            checkpoints_ok = all(os.path.exists(data.get("path", cp)) for cp, data in inv.items())
            if checkpoints_ok:
                results["110M_Preservation"] = "PASS"
                print("  ✓ Checkpoint inventory verified (best.pt, best_v2.pt, best_v3.pt intact).")
            else:
                results["110M_Preservation"] = "FAIL"
        else:
            results["110M_Preservation"] = "FAIL"
    except Exception as e:
        results["110M_Preservation"] = f"FAIL ({e})"

    # Step 2: Check OpenBioLLM 8B Provider
    print("[2/10] Auditing OpenBioLLM-8B Ollama Provider...")
    try:
        ob_prov = OllamaModelProvider("richardyoung/openbiollm:latest")
        if ob_prov.is_available():
            ans = ob_prov.generate("What is blood pressure?", max_tokens=30)
            if ans and len(ans) > 5:
                results["OpenBioLLM_8B"] = "PASS"
                print(f"  ✓ OpenBioLLM-8B active. Response: '{ans[:60]}...'")
            else:
                results["OpenBioLLM_8B"] = "FAIL (Empty response)"
        else:
            results["OpenBioLLM_8B"] = "FAIL (Model unavailable in Ollama)"
    except Exception as e:
        results["OpenBioLLM_8B"] = f"FAIL ({e})"

    # Step 3: Check Llama 3.2 3B Fallback Provider
    print("[3/10] Auditing Llama 3.2 3B Ollama Provider...")
    try:
        l3_prov = OllamaModelProvider("llama3.2:3b")
        if l3_prov.is_available():
            ans = l3_prov.generate("Define venipuncture.", max_tokens=30)
            if ans and len(ans) > 5:
                results["Llama32_3B"] = "PASS"
                print(f"  ✓ Llama 3.2 3B active. Response: '{ans[:60]}...'")
            else:
                results["Llama32_3B"] = "FAIL (Empty response)"
        else:
            results["Llama32_3B"] = "FAIL (Model unavailable in Ollama)"
    except Exception as e:
        results["Llama32_3B"] = f"FAIL ({e})"

    # Step 4: Check MedicalTransformerLM 110M PyTorch MPS Provider
    print("[4/10] Auditing MedicalTransformerLM 110M PyTorch Provider...")
    try:
        mt_prov = MedicalTransformerProvider()
        if mt_prov.is_available():
            meta = mt_prov.get_metadata()
            results["110M_PyTorch_Provider"] = "PASS"
            print(f"  ✓ 110M PyTorch provider ready on MPS ({meta['parameters']:,} parameters).")
        else:
            results["110M_PyTorch_Provider"] = "FAIL"
    except Exception as e:
        results["110M_PyTorch_Provider"] = f"FAIL ({e})"

    # Step 5: Check RAG V2 Vector Database & Hybrid Retriever
    print("[5/10] Auditing RAG V2 Database & Retriever...")
    try:
        pipeline = MedicalRAGPipeline()
        hits = pipeline.hybrid_retriever.search("venipuncture cleaning alcohol", top_k=2)
        if hits and len(hits) > 0:
            top_chunk, score = hits[0]
            results["RAG_Retriever"] = "PASS"
            print(f"  ✓ RAG Retriever active. Top Score: {score:.4f} | Chunk ID: {top_chunk.get('chunk_id')}")
        else:
            results["RAG_Retriever"] = "FAIL"
    except Exception as e:
        results["RAG_Retriever"] = f"FAIL ({e})"

    # Step 6: Check Intent Router Layer
    print("[6/10] Auditing Intent Router Layer...")
    try:
        intent_next = classify_intent("What should I do next?")
        vr_ans = format_deterministic_vr_response(intent_next, current_step=11, step_name="Insert Tube", last_mistake=None)
        if intent_next == "NEXT_STEP" and vr_ans and "Insert the" in vr_ans["answer"]:
            results["Intent_Router"] = "PASS"
            print(f"  ✓ Intent Router deterministic response verified ({vr_ans['answer']}).")
        else:
            results["Intent_Router"] = "FAIL"
    except Exception as e:
        results["Intent_Router"] = f"FAIL ({e})"

    # Step 7: Check Grounding Checker Guardrail
    print("[7/10] Auditing Grounding Checker Guardrail...")
    try:
        dummy_chunks = [{"text": "Venipuncture site is disinfected with 70% isopropyl alcohol."}]
        valid_ans, ok, conf = validate_answer_grounding("The site is cleaned with 70% alcohol for disinfection.", dummy_chunks, "Why clean site?")
        if ok and valid_ans:
            results["Grounding_Checker"] = "PASS"
            print("  ✓ Grounding Checker guardrail active and validating overlap.")
        else:
            results["Grounding_Checker"] = "FAIL"
    except Exception as e:
        results["Grounding_Checker"] = f"FAIL ({e})"

    # Step 8: Check Multi-Model Router & AUTO Mode
    print("[8/10] Auditing Multi-Model Router & AUTO Mode...")
    try:
        router = ModelRouter()
        p, key, m = router.select_provider("auto")
        if p.is_available() and key in ["openbiollm", "llama32", "medical_transformer_110m"]:
            results["Model_Router"] = "PASS"
            print(f"  ✓ Model Router resolved AUTO to: {key} ({m['model']}).")
        else:
            results["Model_Router"] = "FAIL"
    except Exception as e:
        results["Model_Router"] = f"FAIL ({e})"

    # Step 9: Check Multi-Model Judge & Ensemble Engine
    print("[9/10] Auditing Multi-Model Judge & Ensemble Engine...")
    try:
        pipeline = MedicalRAGPipeline()
        ens_res = pipeline.answer_question_ensemble("Why is hand hygiene important before venipuncture?")
        if ens_res and "winning_model" in ens_res and len(ens_res.get("ensemble_candidates", [])) > 0:
            results["Ensemble_Judge"] = "PASS"
            print(f"  ✓ Ensemble Judge active. Winner: {ens_res['winning_model']} (Score: {ens_res['winning_score']}/100).")
        else:
            results["Ensemble_Judge"] = "FAIL"
    except Exception as e:
        results["Ensemble_Judge"] = f"FAIL ({e})"

    # Step 10: Check FastAPI Live REST API Server
    print("[10/10] Auditing Live FastAPI Server on http://127.0.0.1:8000...")
    try:
        h_res = requests.get("http://127.0.0.1:8000/health", timeout=3).json()
        a_res = requests.post("http://127.0.0.1:8000/ask", json={"question": "What equipment is used for venipuncture?", "model": "openbiollm"}, timeout=20).json()
        if h_res.get("status") == "healthy" and a_res.get("answer"):
            results["FastAPI_Server"] = "PASS"
            print(f"  ✓ FastAPI Server online. Health: {h_res['status']} | Model: {a_res['model']}")
        else:
            results["FastAPI_Server"] = "FAIL"
    except Exception as e:
        results["FastAPI_Server"] = f"FAIL ({e})"

    print("\n============================================================")
    print("                    FINAL AUDIT SUMMARY")
    print("============================================================")
    all_pass = True
    for comp, status in results.items():
        print(f"  {comp:25s}: {status}")
        if "PASS" not in status:
            all_pass = False
    print("============================================================")
    if all_pass:
        print("  SYSTEM STATUS: 100% HEALTHY — ALL 10 MODULES VERIFIED PASS")
    else:
        print("  SYSTEM STATUS: ATTENTION REQUIRED")
    print("============================================================\n")


if __name__ == "__main__":
    run_comprehensive_audit()
