#!/usr/bin/env python3
"""
Phases 4 & 5 Module: Metadata-Aware BM25 Hybrid Retriever (RAG V2 Engine)
Combines BM25 score with query normalization and metadata ranking signals (step, topic, source_id).
Enforces strict stopword filtering, punctuation removal, and zero-keyword boost suppression.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

from rag.normalizer import normalize_query
from rag.database import STOP_WORDS

MEDICAL_SYNONYM_MAP = {
    "cbc": "complete blood count hematology",
    "edta": "ethylenediaminetetraacetic acid purple lavender tube",
    "sst": "serum separator tube gold red clot activator",
    "pst": "plasma separator tube light green heparin",
    "pt": "prothrombin time coagulation light blue sodium citrate",
    "ptt": "partial thromboplastin time coagulation light blue",
    "inr": "international normalized ratio coagulation light blue",
    "iv": "intravenous therapy catheter",
    "clsi": "clinical laboratory standards institute gp41 order of draw",
    "sugar": "glucose diabetes mellitus",
    "bp": "blood pressure hypertension sphygmomanometer",
    "heart attack": "myocardial infarction acute coronary syndrome",
    "draw order": "order of draw tube sequence clsi",
    "needle angle": "insertion angle 15 to 30 degrees bevel up",
    "tourniquet": "tourniquet constricting band venous distention 1 minute",
}

def expand_medical_query(query: str) -> str:
    """Expands medical abbreviations and colloquial terms into formal clinical terminology for higher retrieval recall."""
    if not query:
        return ""
    expanded = query
    q_lower = query.lower()
    for acronym, formal in MEDICAL_SYNONYM_MAP.items():
        pattern = r'\b' + re.escape(acronym) + r'\b'
        if re.search(pattern, q_lower):
            expanded += " " + formal
    return expanded

def tokenize_clean_text(text: str) -> List[str]:
    """
    Cleans text, strips punctuation, converts to lowercase, and filters out stop words.
    Preserves single-digit step numbers (e.g. '0'..'9').
    """
    if not text:
        return []
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return [w for w in words if w not in STOP_WORDS and (len(w) > 1 or w.isdigit())]

class MetadataAwareHybridRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_words_list = [tokenize_clean_text(c.get("text", "")) for c in chunks]
        self.doc_len = [len(w) for w in self.doc_words_list]
        self.avgdl = sum(self.doc_len) / max(len(chunks), 1)
        self.doc_freqs = Counter()
        self.N = len(chunks)
        
        for words in self.doc_words_list:
            for w in set(words):
                self.doc_freqs[w] += 1

    def compute_bm25_score(self, query_words: List[str], chunk_idx: int) -> float:
        if not query_words:
            return 0.0
            
        doc_words = self.doc_words_list[chunk_idx]
        if not doc_words:
            return 0.0
            
        word_counts = Counter(doc_words)
        score = 0.0
        
        for qw in query_words:
            if qw in word_counts:
                df = self.doc_freqs.get(qw, 0)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                tf = word_counts[qw]
                len_norm = 1.0 - self.b + self.b * (self.doc_len[chunk_idx] / self.avgdl)
                score += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
        return score

    def compute_metadata_boost(self, chunk: Dict[str, Any], query_text: str, current_step: Optional[int] = None) -> float:
        boost = 0.0
        q_lower = query_text.lower()
        
        # 1. Step Metadata Match
        c_step = chunk.get("step")
        if current_step is not None and c_step == current_step:
            boost += 1.5
            
        step_match = re.search(r"step\s*(\d+)", q_lower)
        if step_match:
            target_step = int(step_match.group(1))
            if c_step == target_step:
                boost += 2.0
                
        # 2. Specific Topic Match (must be non-generic)
        c_topic = str(chunk.get("topic", "")).strip().lower()
        if c_topic and len(c_topic) > 3 and c_topic not in ["general", "phlebotomy", "venipuncture"] and c_topic in q_lower:
            boost += 1.0
            
        return boost

    def search(
        self,
        query: str,
        top_k: int = 5,
        current_step: Optional[int] = None,
        use_normalization: bool = True,
        use_metadata: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        
        processed_query = normalize_query(query) if use_normalization else query
        expanded_query = expand_medical_query(processed_query)
        q_words = tokenize_clean_text(expanded_query)
        
        if not q_words:
            # Fallback to raw query tokens if normalized query produced no content words
            q_words = tokenize_clean_text(expand_medical_query(query))
            
        scored_results = []
        for idx, chunk in enumerate(self.chunks):
            bm25_score = self.compute_bm25_score(q_words, idx)
            
            # Suppress metadata boost if zero content keywords matched
            meta_boost = 0.0
            if bm25_score > 0.0 and use_metadata:
                meta_boost = self.compute_metadata_boost(chunk, query, current_step)
                
            hybrid_score = bm25_score + meta_boost
            
            # Preserve chunk_id metadata on every chunk
            if "chunk_id" not in chunk:
                chunk["chunk_id"] = chunk.get("id", f"doc_{chunk.get('source','src')}_chunk_{idx}")
                
            scored_results.append((chunk, hybrid_score))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
