#!/usr/bin/env python3
"""
Phase 13 Module: Multi-Model Local Medical RAG Pipeline V2
Combines Local Vector Database Retrieval with Multi-Model Provider Routing:
  1. Production Candidate: OpenBioLLM-8B (richardyoung/openbiollm:latest) via Ollama
  2. Benchmark / Fallback: Llama 3.2 3B (llama3.2:3b) via Ollama
  3. Research / Offline: PyTorch MedicalTransformerLM (~110.04M Parameters) on MPS
100% Local Execution on Apple Silicon. Zero cloud APIs required.
"""

import os
import sys
import re
import torch
import time
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.model_provider import ModelRouter
from rag.retriever import MedicalRetriever
from rag.retriever_v2 import MetadataAwareHybridRetriever
from rag.cache import medical_cache
from scripts.validate_grounding import validate_answer_grounding

DISCLAIMER_TEXT = (
    "This system is an educational/research prototype and is not a substitute "
    "for professional medical advice, diagnosis, or treatment."
)

SAFE_REFUSAL_TEXT = "I don't have enough verified information in the current knowledge base to answer that reliably."


def sanitize_response_text(text: str) -> str:
    """
    Sanitizes generated text by removing internal special tokens, instruction headers,
    markdown section headers, and prompt artifacts before returning to UI/Unity.
    Also extracts the answer from echoed Ollama completions.
    """
    if not text:
        return ""
    # 0. Extract answer from echoed RAG prompt (Ollama /api/generate echoes prompt)
    for split_key in ["\nAnswer:", "\nYour Answer:", "\nAnswer :", "ANSWER:"]:
        if split_key in text:
            text = text.split(split_key)[-1].strip()
            break
    if text.startswith("Answer:"):
        text = text[len("Answer:"):].strip()
    # 0b. Strip echoed Q&A chunk headers like "What is venipuncture? Answer: " at start
    text = re.sub(r'^[^.!?]*\?\s*(Answer|A)\s*:\s*', '', text, flags=re.IGNORECASE).strip()
    # 0c. Remove leading "Context Evidence:..." blocks if present
    if "Context Evidence:" in text:
        parts = text.split("\n\n")
        if len(parts) > 1:
            text = parts[-1].strip()
    # 0d. Strip echoed leading question sentences like "What is venipuncture? " at beginning of response
    text = re.sub(r'^(What|How|Why|Which|When|Where|Can|Is|Are|Explain|Describe)[^.!?\n]*\?\s*', '', text, flags=re.IGNORECASE).strip()
    # 1. Remove special tokens
    text = re.sub(r"<\|(endoftext|eos|bos|pad|unk)\|>", "", text)
    # 2. Remove markdown step/section headers (e.g. "## Step 1: ...")
    text = re.sub(r"##\s*Step\s*\d+:[^\.\n]*[\.]?", "", text, flags=re.IGNORECASE)
    # 3. Remove template instruction headers & prompt fragments
    text = re.sub(r"###\s*(Instruction|Response|System):[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Instruction:[^\n]*|Medical Answer:\s*|Context/Input:\s*|Answer question[^\n]*", "", text, flags=re.IGNORECASE)
    # 4. Remove any remaining "Context Evidence:" or "[CLINICAL]" prefix lines
    text = re.sub(r"Context Evidence:[^\n]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[CLINICAL\][^\n]*\n?", "", text, flags=re.IGNORECASE)
    # 5. Remove inline citations, brackets, and reference phrasing from the answer
    text = re.sub(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]', '', text)
    text = re.sub(r'\[\s*\d+\s*-\s*\d+\s*\]', '', text)
    text = re.sub(r'\(\s*(?:Clinical\s+)?Reference\s*(?:\[\d+\]|\d+)?\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:According to|Based on)\s+(?:the\s+)?(?:verified\s+)?(?:clinical\s+)?reference\s*(?:\[\d+\]|\d+)?\s*,?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:According to|Based on)\s+(?:the\s+)?sources?\s*(?:\[\d+\]|\d+)?\s*,?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text



class MedicalRAGPipeline:
    """
    Multi-Model Local Medical RAG Pipeline supporting hybrid BM25 metadata retrieval,
    configurable model selection (OpenBioLLM, Llama 3.2, MedicalTransformer 110M),
    and post-generation grounding validation.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_v3.pt",
        tokenizer_path: str = "tokenizer/artifacts/tokenizer.json",
        db_dir: str = "data/rag_db",
        device_name: str = "mps"
    ):
        self.retriever = MedicalRetriever(db_dir=db_dir)
        self.hybrid_retriever = MetadataAwareHybridRetriever(chunks=self.retriever.db.chunks)
        self.router = ModelRouter()

    def answer_question(
        self,
        question: str,
        top_k: int = 4,
        max_new_tokens: int = 120,
        temperature: float = 0.3,
        top_k_sampling: int = 40,
        top_p: float = 0.9,
        relevance_threshold: float = 0.01,
        model: str = "openbiollm",
        current_step: Optional[int] = None,
        step_name: Optional[str] = None,
        last_mistake: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes RAG pipeline for the requested model key or AUTO mode.
        """
        # 0. Sub-millisecond Exact/Normalized Query Cache Check
        if str(model).lower().strip() not in ["ensemble", "all", "judge", "multi_model"]:
            cached = medical_cache.get(question, model=model)
            if cached:
                return cached

        # Route to Unified Collaborative Pipeline if requested
        if str(model).lower().strip() in ["unified", "collaborative", "combined", "all_together", "rag_openbio_llama"]:
            return self.answer_question_unified(
                question=question,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                current_step=current_step,
                step_name=step_name,
                last_mistake=last_mistake
            )

        # 1. RAG V2 Hybrid Search
        scored_hits = self.hybrid_retriever.search(
            query=question,
            top_k=top_k,
            current_step=current_step,
            use_normalization=True,
            use_metadata=True
        )
        chunks = [h[0] for h in scored_hits]

        # 2. Build Source Metadata & Chunk Provenance
        sources = []
        clean_snippets = []
        top_score = scored_hits[0][1] if scored_hits else 0.0

        for idx, (chunk, score) in enumerate(scored_hits):
            cid = chunk.get("chunk_id", chunk.get("id", f"chunk_{idx}"))
            sources.append({
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "score": round(score, 4),
                "topic": chunk.get("topic", "Venipuncture Guidelines"),
                "step": chunk.get("step", None),
                "source_id": chunk.get("source", "SRC_CLSI_01"),
                "source_title": chunk.get("topic", "Venipuncture Guidelines"),
                "source_section": chunk.get("source_section", "Clinical Standard"),
                "source_page": chunk.get("source_page", "N/A"),
                "source_url": chunk.get("url", "https://clsi.org/standards/products/methodology/documents/gp41/"),
                "relevance_score": round(score, 4),
                "snippet": chunk["text"][:150] + "..."
            })
            txt = re.sub(r'Instruction:[^\n]+\n?|Context/Input:\s*|Medical Details:\s*', '', chunk['text']).strip()
            if txt:
                clean_snippets.append(txt)

        rag_context = "\n\n".join(clean_snippets)  # Use ALL retrieved chunks, not just first 2

        # Select Provider and get Metadata upfront
        provider_obj, resolved_key, meta = self.router.select_provider(model)

        # 3. Relevance Thresholding Check
        if top_score < relevance_threshold or not clean_snippets:
            res = {
                "question": question,
                "answer": SAFE_REFUSAL_TEXT,
                "provider": meta.get("provider", "system"),
                "model": meta.get("model", resolved_key),
                "rag": True,
                "grounded": True,
                "confidence": "high",
                "sources": sources
            }
            if meta.get("parameters"):
                res["parameters"] = meta.get("parameters")
            return res

        # 4. Formulate Prompt based on Provider Type
        if meta.get("provider") == "pytorch":
            prompt = f"Instruction: Use evidence to answer: {question}\nContext: {rag_context[:800]}\nMedical Answer:"
            system_prompt = None
        else:
            system_prompt = (
                "You are OpenBioLLM, an expert clinical AI assistant. Answer the user's medical question "
                "clearly, accurately, and thoroughly using the provided verified clinical reference. "
                "Do NOT include citations, brackets, reference numbers (such as [1] or [2]), or phrases like 'According to Reference'. "
                "Do NOT repeat the reference, prompt headers, or question. Begin your answer directly."
            )
            prompt = f"Clinical Reference:\n{rag_context}\n\nQuestion: {question}"

        # 5. Generate Response via Selected Provider
        try:
            raw_ans = provider_obj.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k_sampling,
                top_p=top_p
            )
            if meta.get("provider") == "pytorch" and "Medical Answer:" in raw_ans:
                raw_ans = raw_ans.split("Medical Answer:")[-1].strip()
        except Exception as e:
            # Fallback if primary generation fails
            raw_ans = SAFE_REFUSAL_TEXT

        sanitized_ans = sanitize_response_text(raw_ans)
        final_ans, is_grounded, conf = validate_answer_grounding(sanitized_ans, chunks, question)
        final_ans = sanitize_response_text(final_ans)

        # Handle Ensemble / Multi-Model Judge Mode
        if str(model).lower().strip() in ["ensemble", "all", "judge", "multi_model"]:
            return self.answer_question_ensemble(
                question=question,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k_sampling=top_k_sampling,
                top_p=top_p,
                relevance_threshold=relevance_threshold,
                current_step=current_step,
                step_name=step_name,
                last_mistake=last_mistake
            )

        res = {
            "question": question,
            "answer": final_ans,
            "provider": meta.get("provider", "ollama"),
            "model": meta.get("model", resolved_key),
            "rag": True,
            "grounded": is_grounded,
            "confidence": conf,
            "sources": sources
        }
        if meta.get("parameters"):
            res["parameters"] = meta.get("parameters")

        # Save to Cache if grounded and non-empty
        if is_grounded and final_ans and final_ans != SAFE_REFUSAL_TEXT:
            medical_cache.set(question, res, model=model)

        return res

    def answer_question_stream(
        self,
        question: str,
        top_k: int = 4,
        max_new_tokens: int = 250,
        temperature: float = 0.3,
        model: str = "openbiollm",
        current_step: Optional[int] = None,
        step_name: Optional[str] = None,
        last_mistake: Optional[str] = None
    ):
        """
        Streams RAG response: yields initial metadata, tokens in real time, and final completion.
        """
        # Route to Unified Collaborative Pipeline if requested
        if str(model).lower().strip() in ["unified", "collaborative", "combined", "all_together", "rag_openbio_llama"]:
            for evt in self.answer_question_unified_stream(
                question=question,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                current_step=current_step,
                step_name=step_name,
                last_mistake=last_mistake
            ):
                yield evt
            return

        t0 = time.time()
        # 1. Check cache first
        cached = medical_cache.get(question, model=model)
        if cached:
            yield {"type": "meta", "intent": cached.get("intent", "CLINICAL_QA"), "sources": cached.get("sources", []), "cache_hit": True}
            yield {"type": "token", "delta": cached.get("answer", "")}
            yield {
                "type": "done",
                "answer": cached.get("answer", ""),
                "grounded": cached.get("grounded", True),
                "confidence": cached.get("confidence", "high"),
                "total_ms": round((time.time() - t0) * 1000, 2),
                "cache_hit": True
            }
            return

        # 2. Hybrid retrieval
        scored_hits = self.hybrid_retriever.search(
            query=question,
            top_k=top_k,
            current_step=current_step,
            use_normalization=True,
            use_metadata=True
        )
        chunks = [h[0] for h in scored_hits]
        sources = []
        clean_snippets = []
        top_score = scored_hits[0][1] if scored_hits else 0.0

        for idx, (chunk, score) in enumerate(scored_hits):
            cid = chunk.get("chunk_id", chunk.get("id", f"chunk_{idx}"))
            sources.append({
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "score": round(score, 4),
                "topic": chunk.get("topic", "Venipuncture Guidelines"),
                "step": chunk.get("step", None),
                "source_id": chunk.get("source", "SRC_CLSI_01"),
                "source_title": chunk.get("topic", "Venipuncture Guidelines"),
                "source_section": chunk.get("source_section", "Clinical Standard"),
                "source_page": chunk.get("source_page", "N/A"),
                "source_url": chunk.get("url", "https://clsi.org/standards/products/methodology/documents/gp41/"),
                "relevance_score": round(score, 4),
                "snippet": chunk["text"][:150] + "..."
            })
            txt = re.sub(r'Instruction:[^\n]+\n?|Context/Input:\s*|Medical Details:\s*', '', chunk['text']).strip()
            if txt:
                clean_snippets.append(txt)

        yield {"type": "meta", "intent": "CLINICAL_QA", "sources": sources, "cache_hit": False}

        if top_score < 0.01 or not clean_snippets:
            yield {"type": "token", "delta": SAFE_REFUSAL_TEXT}
            yield {
                "type": "done",
                "answer": SAFE_REFUSAL_TEXT,
                "grounded": True,
                "confidence": "high",
                "total_ms": round((time.time() - t0) * 1000, 2),
                "cache_hit": False
            }
            return

        rag_context = "\n\n".join(clean_snippets)
        provider_obj, resolved_key, meta = self.router.select_provider(model)

        if meta.get("provider") == "pytorch":
            prompt = f"Instruction: Use evidence to answer: {question}\nContext: {rag_context[:800]}\nMedical Answer:"
            system_prompt = None
        else:
            system_prompt = (
                "You are OpenBioLLM, an expert clinical AI assistant. Answer the user's medical question "
                "clearly, accurately, and thoroughly using the provided verified clinical reference. "
                "Do NOT include citations, brackets, reference numbers (such as [1] or [2]), or phrases like 'According to Reference'. "
                "Do NOT repeat the reference, prompt headers, or question. Begin your answer directly."
            )
            prompt = f"Clinical Reference:\n{rag_context}\n\nQuestion: {question}"

        full_raw = []
        for delta in provider_obj.generate_stream(prompt=prompt, system_prompt=system_prompt, max_tokens=max_new_tokens, temperature=temperature):
            full_raw.append(delta)
            yield {"type": "token", "delta": delta}

        raw_str = "".join(full_raw)
        sanitized = sanitize_response_text(raw_str)
        final_ans, is_grounded, conf = validate_answer_grounding(sanitized, chunks, question)
        final_ans = sanitize_response_text(final_ans)

        # Cache result
        res_data = {
            "question": question,
            "answer": final_ans,
            "provider": meta.get("provider", "ollama"),
            "model": meta.get("model", resolved_key),
            "rag": True,
            "grounded": is_grounded,
            "confidence": conf,
            "sources": sources
        }
        if is_grounded and final_ans and final_ans != SAFE_REFUSAL_TEXT:
            medical_cache.set(question, res_data, model=model)

        yield {
            "type": "done",
            "answer": final_ans,
            "grounded": is_grounded,
            "confidence": conf,
            "total_ms": round((time.time() - t0) * 1000, 2),
            "cache_hit": False
        }

    def evaluate_candidate_score(self, candidate_ans: str, chunks: List[Dict[str, Any]], model_key: str, is_grounded: bool) -> float:
        if candidate_ans == SAFE_REFUSAL_TEXT or not candidate_ans.strip():
            return 0.0

        score = 0.0
        # 1. Base Model Capability Weight
        if model_key in ["openbiollm", "openbiollm-8b"]:
            score += 35.0
        elif model_key in ["llama32", "llama3.2"]:
            score += 25.0
        elif model_key in ["medical_transformer_110m", "medical_transformer"]:
            score += 15.0

        # 2. Grounded Status
        if is_grounded:
            score += 30.0

        # 3. Evidence Overlap
        combined_evidence = " ".join([c.get("text", "").lower() for c in chunks])
        words = [w for w in candidate_ans.lower().split() if len(w) > 4]
        if words:
            matches = sum(1 for w in words if w in combined_evidence)
            ratio = matches / len(words)
            score += min(ratio * 25.0, 25.0)

        # 4. Length / Completeness Penalty/Bonus
        word_count = len(candidate_ans.split())
        if 15 <= word_count <= 80:
            score += 10.0
        elif word_count < 8:
            score -= 10.0

        return round(max(0.0, score), 2)

    def answer_question_ensemble(
        self,
        question: str,
        top_k: int = 2,
        max_new_tokens: int = 120,
        temperature: float = 0.3,
        top_k_sampling: int = 40,
        top_p: float = 0.9,
        relevance_threshold: float = 0.01,
        current_step: Optional[int] = None,
        step_name: Optional[str] = None,
        last_mistake: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes query against ALL THREE models (OpenBioLLM-8B, Llama 3.2 3B, MedicalTransformerLM 110M),
        evaluates candidate answers using the Judge scorer, and finalizes the best answer.
        """
        # 1. Retrieve Evidence Chunks via RAG V2 Hybrid Search
        scored_hits = self.hybrid_retriever.search(
            query=question,
            top_k=top_k,
            current_step=current_step,
            use_normalization=True,
            use_metadata=True
        )
        sources = []
        clean_snippets = []
        top_score = scored_hits[0][1] if scored_hits else 0.0

        for idx, (chunk, score) in enumerate(scored_hits):
            cid = chunk.get("chunk_id", chunk.get("id", f"chunk_{idx}"))
            sources.append({
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "score": round(score, 4),
                "topic": chunk.get("topic", "Venipuncture Guidelines"),
                "step": chunk.get("step", None),
                "source_id": chunk.get("source", "SRC_CLSI_01"),
                "source_title": chunk.get("topic", "Venipuncture Guidelines"),
                "source_section": chunk.get("source_section", "Clinical Standard"),
                "source_page": chunk.get("source_page", "N/A"),
                "source_url": chunk.get("url", "https://clsi.org/standards/products/methodology/documents/gp41/"),
                "relevance_score": round(score, 4),
                "snippet": chunk["text"][:150] + "..."
            })
            txt = re.sub(r'Instruction:[^\n]+\n?|Context/Input:\s*|Medical Details:\s*', '', chunk['text']).strip()
            if txt:
                clean_snippets.append(txt)

        rag_context = "\n\n".join(clean_snippets[:2]) if clean_snippets else ""

        if top_score < relevance_threshold or not clean_snippets:
            return {
                "question": question,
                "answer": SAFE_REFUSAL_TEXT,
                "provider": "ensemble_judge",
                "model": "ensemble_judge",
                "rag": True,
                "grounded": True,
                "confidence": "high",
                "sources": sources,
                "winning_model": "None (Safety Refusal)",
                "ensemble_candidates": []
            }

        # 3. Query All 3 Models
        models_to_query = [
            ("openbiollm", "OpenBioLLM 8B"),
            ("llama32", "Llama 3.2 3B"),
            ("medical_transformer_110m", "MedicalTransformerLM 110M")
        ]

        candidates = []
        best_candidate = None
        highest_score = -1.0

        for m_key, m_label in models_to_query:
            try:
                p_obj, r_key, p_meta = self.router.select_provider(m_key)
                if not p_obj.is_available():
                    continue

                if p_meta.get("provider") == "pytorch":
                    p_prompt = f"Instruction: Use evidence to answer: {question}\nContext: {rag_context[:300]}\nMedical Answer:"
                else:
                    p_prompt = (
                        f"Context Evidence:\n{rag_context}\n\n"
                        f"Question: {question}\n\n"
                        "Instructions: Answer the medical question accurately using only the evidence context provided above in 2-3 clear sentences.\n\n"
                        "Answer:"
                    )

                t0 = time.time()
                raw_out = p_obj.generate(prompt=p_prompt, max_tokens=max_new_tokens, temperature=temperature)
                lat_ms = round((time.time() - t0) * 1000, 2)

                if p_meta.get("provider") == "pytorch" and "Medical Answer:" in raw_out:
                    raw_out = raw_out.split("Medical Answer:")[-1].strip()

                s_out = sanitize_response_text(raw_out)
                chunks_list = [h[0] for h in scored_hits]
                final_c_ans, c_grounded, c_conf = validate_answer_grounding(s_out, chunks_list, question)
                final_c_ans = sanitize_response_text(final_c_ans)

                score_val = self.evaluate_candidate_score(final_c_ans, chunks_list, m_key, c_grounded)

                cand_record = {
                    "model_key": m_key,
                    "model_name": p_meta.get("model", m_label),
                    "model_label": m_label,
                    "provider": p_meta.get("provider"),
                    "answer": final_c_ans,
                    "grounded": c_grounded,
                    "confidence": c_conf,
                    "score": score_val,
                    "latency_ms": lat_ms
                }
                candidates.append(cand_record)

                if score_val > highest_score:
                    highest_score = score_val
                    best_candidate = cand_record

            except Exception as e:
                print(f"[!] Ensemble candidate error for {m_key}: {e}")

        if not best_candidate:
            best_candidate = {
                "model_key": "openbiollm",
                "model_name": "richardyoung/openbiollm:latest",
                "model_label": "OpenBioLLM 8B",
                "provider": "ollama",
                "answer": SAFE_REFUSAL_TEXT,
                "grounded": True,
                "confidence": "high",
                "score": 0.0,
                "latency_ms": 0.0
            }

        return {
            "question": question,
            "answer": best_candidate["answer"],
            "provider": best_candidate["provider"],
            "model": best_candidate["model_name"],
            "winning_model": f"{best_candidate['model_label']} ({best_candidate['model_name']})",
            "winning_score": best_candidate["score"],
            "rag": True,
            "grounded": best_candidate["grounded"],
            "confidence": best_candidate["confidence"],
            "sources": sources,
            "ensemble_candidates": candidates
        }

    def answer_question_unified(
        self,
        question: str,
        top_k: int = 4,
        max_new_tokens: int = 250,
        temperature: float = 0.3,
        current_step: Optional[int] = None,
        step_name: Optional[str] = None,
        last_mistake: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified Collaborative Pipeline:
        Combines Hybrid RAG + OpenBioLLM (Rapid Clinical Extraction) + Llama 3.2 (Synthesis with Citations)
        into a single, cohesive, grounded medical answer.
        """
        t0 = time.time()
        # 0. Check cache
        cached = medical_cache.get(question, model="unified")
        if cached:
            return cached

        # 1. RAG V2 Hybrid Search
        scored_hits = self.hybrid_retriever.search(
            query=question,
            top_k=top_k,
            current_step=current_step,
            use_normalization=True,
            use_metadata=True
        )
        chunks = [h[0] for h in scored_hits]
        sources = []
        clean_snippets = []
        top_score = scored_hits[0][1] if scored_hits else 0.0

        for idx, (chunk, score) in enumerate(scored_hits):
            cid = chunk.get("chunk_id", chunk.get("id", f"chunk_{idx}"))
            sources.append({
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "score": round(score, 4),
                "topic": chunk.get("topic", "Venipuncture Guidelines"),
                "step": chunk.get("step", None),
                "source_id": chunk.get("source", "SRC_CLSI_01"),
                "source_title": chunk.get("topic", "Venipuncture Guidelines"),
                "source_section": chunk.get("source_section", "Clinical Standard"),
                "source_page": chunk.get("source_page", "N/A"),
                "source_url": chunk.get("url", "https://clsi.org/standards/products/methodology/documents/gp41/"),
                "relevance_score": round(score, 4),
                "snippet": chunk["text"][:150] + "..."
            })
            txt = re.sub(r'Instruction:[^\n]+\n?|Context/Input:\s*|Medical Details:\s*', '', chunk['text']).strip()
            if txt:
                clean_snippets.append(f"[{idx+1}] {txt}")

        if top_score < 0.01 or not clean_snippets:
            return {
                "question": question,
                "answer": SAFE_REFUSAL_TEXT,
                "provider": "unified_collaborative",
                "model": "Unified (RAG + OpenBioLLM + Llama 3.2)",
                "rag": True,
                "grounded": True,
                "confidence": "high",
                "sources": sources
            }

        rag_context = "\n\n".join(clean_snippets)

        # 2. Stage 2: OpenBioLLM (Rapid Biomedical Clinical Extraction)
        bio_provider, _, _ = self.router.select_provider("openbiollm")
        llama_provider, _, _ = self.router.select_provider("llama32")

        bio_extract = ""
        try:
            bio_sys = (
                "You are OpenBioLLM, a specialized biomedical clinical AI. "
                "Extract the key clinical facts, medical steps, and anatomy from the reference to answer the question. "
                "Keep it concise in 1-3 clinical sentences. Do NOT repeat the question or prompt."
            )
            bio_user = f"Clinical Reference:\n{rag_context}\n\nQuestion: {question}"
            bio_extract = bio_provider.generate(
                prompt=bio_user,
                system_prompt=bio_sys,
                max_tokens=80,
                temperature=0.2
            )
            bio_extract = sanitize_response_text(bio_extract)
        except Exception as e:
            print(f"[!] OpenBioLLM extraction fallback: {e}")
            bio_extract = rag_context[:300]

        # 3. Stage 3: Llama 3.2 (Final Structuring & Synthesis Engine with Inline Citations)
        try:
            llama_sys = (
                "You are an expert clinical synthesis assistant. Using the verified clinical reference and "
                "OpenBioLLM's clinical findings, produce a clear, authoritative, and complete medical answer in direct clinical prose. "
                "Do NOT include citations, brackets, reference numbers (such as [1] or [2]), or phrases like 'According to Clinical Reference'. "
                "Do NOT echo the prompt or question. Begin your answer directly."
            )
            llama_user = (
                f"Verified Clinical Reference:\n{rag_context}\n\n"
                f"OpenBioLLM Clinical Findings:\n{bio_extract}\n\n"
                f"Question: {question}"
            )
            raw_synthesis = llama_provider.generate(
                prompt=llama_user,
                system_prompt=llama_sys,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
        except Exception as e:
            print(f"[!] Llama synthesis fallback: {e}")
            raw_synthesis = bio_extract

        # 4. Stage 4: Grounding & Conflict Safety Check
        sanitized = sanitize_response_text(raw_synthesis)
        final_ans, is_grounded, conf = validate_answer_grounding(sanitized, chunks, question)
        final_ans = sanitize_response_text(final_ans)

        res = {
            "question": question,
            "answer": final_ans,
            "provider": "unified_collaborative",
            "model": "Unified (RAG + OpenBioLLM + Llama 3.2)",
            "rag": True,
            "grounded": is_grounded,
            "confidence": conf,
            "sources": sources,
            "collaboration": {
                "rag_chunks_used": len(clean_snippets),
                "biomedical_extractor": "OpenBioLLM 8B",
                "synthesis_engine": "Llama 3.2 3B",
                "grounding_check": "Verified" if is_grounded else "Safety Fallback"
            }
        }

        if is_grounded and final_ans and final_ans != SAFE_REFUSAL_TEXT:
            medical_cache.set(question, res, model="unified")

        return res

    def answer_question_unified_stream(
        self,
        question: str,
        top_k: int = 4,
        max_new_tokens: int = 250,
        temperature: float = 0.3,
        current_step: Optional[int] = None,
        step_name: Optional[str] = None,
        last_mistake: Optional[str] = None
    ):
        """
        Streams the Unified Collaborative Pipeline:
        Extracts via OpenBioLLM and streams final synthesized answer via Llama 3.2 in real time.
        """
        t0 = time.time()
        # 0. Check cache first
        cached = medical_cache.get(question, model="unified")
        if cached:
            yield {"type": "meta", "intent": cached.get("intent", "CLINICAL_QA"), "sources": cached.get("sources", []), "cache_hit": True, "model": "Unified (RAG + OpenBioLLM + Llama 3.2)"}
            yield {"type": "token", "delta": cached.get("answer", "")}
            yield {
                "type": "done",
                "answer": cached.get("answer", ""),
                "grounded": cached.get("grounded", True),
                "confidence": cached.get("confidence", "high"),
                "total_ms": round((time.time() - t0) * 1000, 2),
                "cache_hit": True
            }
            return

        # 1. RAG V2 Hybrid Search
        scored_hits = self.hybrid_retriever.search(
            query=question,
            top_k=top_k,
            current_step=current_step,
            use_normalization=True,
            use_metadata=True
        )
        chunks = [h[0] for h in scored_hits]
        sources = []
        clean_snippets = []
        top_score = scored_hits[0][1] if scored_hits else 0.0

        for idx, (chunk, score) in enumerate(scored_hits):
            cid = chunk.get("chunk_id", chunk.get("id", f"chunk_{idx}"))
            sources.append({
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "score": round(score, 4),
                "topic": chunk.get("topic", "Venipuncture Guidelines"),
                "step": chunk.get("step", None),
                "source_id": chunk.get("source", "SRC_CLSI_01"),
                "source_title": chunk.get("topic", "Venipuncture Guidelines"),
                "source_section": chunk.get("source_section", "Clinical Standard"),
                "source_page": chunk.get("source_page", "N/A"),
                "source_url": chunk.get("url", "https://clsi.org/standards/products/methodology/documents/gp41/"),
                "relevance_score": round(score, 4),
                "snippet": chunk["text"][:150] + "..."
            })
            txt = re.sub(r'Instruction:[^\n]+\n?|Context/Input:\s*|Medical Details:\s*', '', chunk['text']).strip()
            if txt:
                clean_snippets.append(f"[{idx+1}] {txt}")

        yield {"type": "meta", "intent": "CLINICAL_QA", "sources": sources, "cache_hit": False, "model": "Unified (RAG + OpenBioLLM + Llama 3.2)"}

        if top_score < 0.01 or not clean_snippets:
            yield {"type": "token", "delta": SAFE_REFUSAL_TEXT}
            yield {
                "type": "done",
                "answer": SAFE_REFUSAL_TEXT,
                "grounded": True,
                "confidence": "high",
                "total_ms": round((time.time() - t0) * 1000, 2),
                "cache_hit": False
            }
            return

        rag_context = "\n\n".join(clean_snippets)

        # 2. Stage 2: OpenBioLLM (Rapid Clinical Bullet Extraction)
        bio_provider, _, _ = self.router.select_provider("openbiollm")
        llama_provider, _, _ = self.router.select_provider("llama32")

        bio_extract = ""
        try:
            bio_sys = (
                "You are OpenBioLLM, a specialized biomedical clinical AI. "
                "Extract key clinical facts, medical steps, and anatomy from the reference to answer the question. "
                "Keep it concise in 1-3 clinical sentences. Do NOT repeat the question or prompt."
            )
            bio_user = f"Clinical Reference:\n{rag_context}\n\nQuestion: {question}"
            bio_extract = bio_provider.generate(
                prompt=bio_user,
                system_prompt=bio_sys,
                max_tokens=80,
                temperature=0.2
            )
            bio_extract = sanitize_response_text(bio_extract)
        except Exception as e:
            bio_extract = rag_context[:300]

        # 3. Stage 3: Llama 3.2 Stream Synthesis with Inline Citations
        llama_sys = (
            "You are an expert clinical synthesis assistant. Using the verified clinical reference and "
            "OpenBioLLM's clinical findings, produce a clear, authoritative, and complete medical answer in direct clinical prose. "
            "Do NOT include citations, brackets, reference numbers (such as [1] or [2]), or phrases like 'According to Clinical Reference'. "
            "Do NOT echo the prompt or question. Begin your answer directly."
        )
        llama_user = (
            f"Verified Clinical Reference:\n{rag_context}\n\n"
            f"OpenBioLLM Clinical Findings:\n{bio_extract}\n\n"
            f"Question: {question}"
        )

        full_raw = []
        try:
            for delta in llama_provider.generate_stream(
                prompt=llama_user,
                system_prompt=llama_sys,
                max_tokens=max_new_tokens,
                temperature=temperature
            ):
                full_raw.append(delta)
                clean_delta = re.sub(r'\[\s*\d+\s*\]', '', delta)
                if clean_delta:
                    yield {"type": "token", "delta": clean_delta}
        except Exception as e:
            yield {"type": "token", "delta": bio_extract}
            full_raw.append(bio_extract)

        raw_str = "".join(full_raw)
        sanitized = sanitize_response_text(raw_str)
        final_ans, is_grounded, conf = validate_answer_grounding(sanitized, chunks, question)
        final_ans = sanitize_response_text(final_ans)

        res_data = {
            "question": question,
            "answer": final_ans,
            "provider": "unified_collaborative",
            "model": "Unified (RAG + OpenBioLLM + Llama 3.2)",
            "rag": True,
            "grounded": is_grounded,
            "confidence": conf,
            "sources": sources
        }
        if is_grounded and final_ans and final_ans != SAFE_REFUSAL_TEXT:
            medical_cache.set(question, res_data, model="unified")

        yield {
            "type": "done",
            "answer": final_ans,
            "grounded": is_grounded,
            "confidence": conf,
            "total_ms": round((time.time() - t0) * 1000, 2),
            "cache_hit": False
        }


