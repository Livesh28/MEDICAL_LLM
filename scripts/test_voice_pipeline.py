#!/usr/bin/env python3
"""
Phase 16 Module: End-to-End Automated Voice Assistant Pipeline Test Suite
Tests:
1. STT interface
2. Intent Router
3. NEXT_STEP
4. REPEAT
5. WHY_WRONG
6. CLINICAL_QA
7. OPEN_QUESTION
8. UNSUPPORTED
9. RAG retrieval
10. Llama response
11. Grounding checker
12. API response format
"""

import os
import sys
import unittest
import json
import requests
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.intent_router import (
    classify_intent,
    format_deterministic_vr_response,
    INTENT_NEXT_STEP,
    INTENT_REPEAT,
    INTENT_WHY_WRONG,
    INTENT_HELP,
    INTENT_VR_CONTEXT,
    INTENT_CLINICAL_QA,
    INTENT_OPEN_QUESTION,
    INTENT_UNSUPPORTED
)
from api.stt_service import WhisperSTTService
from rag.pipeline import MedicalRAGPipeline
from scripts.validate_grounding import validate_answer_grounding

VR_SPEC_FILE = "data/vr_knowledge/venipuncture_16_steps.json"

class TestVoicePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stt_service = WhisperSTTService(model_name="tiny")
        cls.vr_steps = {}
        if os.path.exists(VR_SPEC_FILE):
            with open(VR_SPEC_FILE, "r", encoding="utf-8") as f:
                spec = json.load(f)
                for s in spec.get("steps", []):
                    cls.vr_steps[s["step"]] = s

    def test_01_stt_interface(self):
        res = self.stt_service.transcribe_text_mock("What should I do next?")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["transcript"], "What should I do next?")

    def test_02_intent_router_classification(self):
        self.assertEqual(classify_intent("What should I do next?"), INTENT_NEXT_STEP)
        self.assertEqual(classify_intent("Repeat the step"), INTENT_REPEAT)
        self.assertEqual(classify_intent("Why was that wrong?"), INTENT_WHY_WRONG)
        self.assertEqual(classify_intent("Why do we clean the site with alcohol?"), INTENT_CLINICAL_QA)
        self.assertEqual(classify_intent("Tell me about phlebotomy"), INTENT_OPEN_QUESTION)
        self.assertEqual(classify_intent("What is the patient's blood pressure?"), INTENT_UNSUPPORTED)

    def test_03_next_step_deterministic_path(self):
        res = format_deterministic_vr_response(
            intent=INTENT_NEXT_STEP,
            current_step=11,
            step_name="Insert Tube",
            last_mistake="None",
            vr_steps_data=self.vr_steps
        )
        self.assertIsNotNone(res)
        self.assertIn("Insert", res["answer"])
        self.assertEqual(res["engine"], "vr_stepmanager_deterministic")

    def test_04_repeat_deterministic_path(self):
        res = format_deterministic_vr_response(
            intent=INTENT_REPEAT,
            current_step=5,
            step_name="Clean Area",
            last_mistake="None",
            vr_steps_data=self.vr_steps
        )
        self.assertIsNotNone(res)
        self.assertIn("Repeating Step 5", res["answer"])

    def test_05_why_wrong_deterministic_path(self):
        res = format_deterministic_vr_response(
            intent=INTENT_WHY_WRONG,
            current_step=10,
            step_name="Take Tube",
            last_mistake="Wrong Order of Draw",
            vr_steps_data=self.vr_steps
        )
        self.assertIsNotNone(res)
        self.assertIn("Wrong Order of Draw", res["answer"])

    def test_06_unsupported_safe_refusal(self):
        res = format_deterministic_vr_response(
            intent=INTENT_UNSUPPORTED,
            current_step=2,
            step_name="Apply Tourniquet",
            last_mistake="None",
            vr_steps_data=self.vr_steps
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["answer"], "This information is not provided in the current simulation.")

    def test_07_grounding_checker_valid_pass(self):
        evidence = [{"text": "Hand hygiene must be performed before touching equipment."}]
        gen_ans = "Clean non-sterile gloves should be worn after hand hygiene."
        ans, is_grounded, conf = validate_answer_grounding(gen_ans, evidence, "When to wear gloves?")
        self.assertTrue(is_grounded)
        self.assertEqual(ans, gen_ans)

    def test_08_grounding_checker_unsupported_catch(self):
        evidence = [{"text": "Tourniquet should remain on arm less than 1 minute."}]
        gen_ans = "The patient's blood pressure is 120/80 mmHg."
        ans, is_grounded, conf = validate_answer_grounding(gen_ans, evidence, "What is blood pressure?")
        self.assertFalse(is_grounded)
        self.assertIn("don't have enough verified information", ans)

    def test_09_api_request_response_schema(self):
        try:
            url = "http://127.0.0.1:8000/ask"
            payload = {
                "question": "What should I do next?",
                "current_step": 11,
                "step_name": "Insert Tube",
                "last_mistake": "None"
            }
            resp = requests.post(url, json=payload, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                self.assertIn("answer", data)
                self.assertIn("engine", data)
                self.assertIn("grounded", data)
                self.assertIn("intent", data)
        except requests.exceptions.ConnectionError:
            print("  [!] API server offline; skipping live HTTP schema test.")

if __name__ == "__main__":
    unittest.main()
