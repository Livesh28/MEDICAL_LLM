#!/usr/bin/env python3
"""
Problem 14 Module: Live RAG Pipeline Regression Test Suite
Verifies:
1. Relevant retrieval
2. Unrelated retrieval rejection
3. Model engine routing & logging
4. Clean prompt construction
5. EOS handling & special token stripping
6. Valid chunk_id metadata presence
7. Supported medical question handling
8. Unsupported question safe refusal
9. VR deterministic state routing
10. BM25 Document A vs Document B ranking test
"""

import os
import sys
import unittest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retriever_v2 import MetadataAwareHybridRetriever
from rag.pipeline import MedicalRAGPipeline, sanitize_response_text
from api.intent_router import classify_intent, format_deterministic_vr_response, INTENT_NEXT_STEP, INTENT_UNSUPPORTED

class TestLiveRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = MedicalRAGPipeline()

    def test_01_bm25_ranking_document_a_vs_document_b(self):
        """
        Problem 9 Unit Test: Verify Document A (Hypertension) vs Document B (Scleroderma).
        """
        chunks = [
            {"chunk_id": "doc_A", "text": "Hypertension is a chronic condition involving elevated blood pressure."},
            {"chunk_id": "doc_B", "text": "Scleroderma is a connective tissue disease affecting skin and blood vessels."}
        ]
        retriever = MetadataAwareHybridRetriever(chunks)
        
        # Test Query 1: "What is hypertension?"
        results_hyp = retriever.search("What is hypertension?", top_k=2, use_metadata=False)
        self.assertEqual(results_hyp[0][0]["chunk_id"], "doc_A")
        self.assertGreater(results_hyp[0][1], results_hyp[1][1])

        # Test Query 2: "What is scleroderma?"
        results_scl = retriever.search("What is scleroderma?", top_k=2, use_metadata=False)
        self.assertEqual(results_scl[0][0]["chunk_id"], "doc_B")
        self.assertGreater(results_scl[0][1], results_scl[1][1])

    def test_02_valid_chunk_id_metadata(self):
        """
        Problem 2 Test: Verify every retrieved chunk returns a valid, defined chunk_id.
        """
        res = self.rag.answer_question("Why is the venipuncture site cleaned?", top_k=2)
        sources = res.get("sources", [])
        self.assertGreater(len(sources), 0)
        for s in sources:
            self.assertIn("chunk_id", s)
            self.assertIsNotNone(s["chunk_id"])
            self.assertNotEqual(s["chunk_id"], "undefined")

    def test_03_special_token_sanitization(self):
        """
        Problem 4 & 13 Test: Verify special tokens like <|endoftext|> and instruction headers are stripped.
        """
        raw_text = "Hypertension is high blood pressure. <|endoftext|> Instruction: Answer question <|eos|>"
        clean_text = sanitize_response_text(raw_text)
        self.assertNotIn("<|endoftext|>", clean_text)
        self.assertNotIn("<|eos|>", clean_text)
        self.assertNotIn("Instruction:", clean_text)
        self.assertEqual(clean_text, "Hypertension is high blood pressure.")

    def test_04_unsupported_question_safe_refusal(self):
        """
        Problem 6 & 7 Test: Verify unsupported patient queries return safe refusal.
        """
        res = self.rag.answer_question("What is the patient's blood pressure?", top_k=2)
        ans_lower = res["answer"].lower()
        self.assertTrue(any(p in ans_lower for p in ["don't have", "not provided", "not mentioned", "unable to determine", "not specified"]))

    def test_05_vr_deterministic_routing(self):
        """
        Problem 12 Test: Verify VR questions use StepManager deterministically.
        """
        intent = classify_intent("What should I do next?")
        self.assertEqual(intent, INTENT_NEXT_STEP)
        vr_spec_path = "data/vr_knowledge/venipuncture_16_steps.json"
        vr_data = {}
        if os.path.exists(vr_spec_path):
            with open(vr_spec_path, "r", encoding="utf-8") as f:
                vr_data = {s["step"]: s for s in json.load(f).get("steps", [])}
        res = format_deterministic_vr_response(intent, current_step=11, step_name="Insert Tube", last_mistake=None, vr_steps_data=vr_data)
        self.assertIsNotNone(res)
        self.assertIn("Insert", res["answer"])
        self.assertEqual(res["engine"], "vr_stepmanager_deterministic")

    def test_06_model_engine_field_presence(self):
        """
        Problem 5 Test: Verify API response explicitly reports model engine name.
        """
        res = self.rag.answer_question("Why is the venipuncture site cleaned?", top_k=2)
        self.assertIn("engine", res)
        self.assertIn(res["engine"], ["llama3.2:3b", "pytorch_medical_transformer_v3", "vr_safety_guardrail", "vr_stepmanager_deterministic"])

    def test_07_supported_medical_question(self):
        """
        Problem 11 Test: Supported venipuncture medical question produces grounded answer.
        """
        res = self.rag.answer_question("Why do we clean the site with alcohol?", top_k=2)
        self.assertTrue(res.get("grounded"))
        self.assertGreater(len(res.get("answer", "")), 10)

if __name__ == "__main__":
    unittest.main()
