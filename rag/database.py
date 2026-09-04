#!/usr/bin/env python3
"""
Phase 13 Module: Local Vector Database with Typo Expansion & MCQ Filtering
Self-contained local vector database with TF-IDF weighting, typo/synonym expansion,
MCQ option filtering, exact keyword matching boost, and cosine similarity search.
"""

import os
import re
import json
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional, Set

# Standard stop words to ignore during vector indexing and search
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "whatever", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "tell", "story", "write", "make", "give", "say", "please", "know", "mean", "define", "explain", "describe", "show", "list"
}

# Typo and clinical spelling correction map
TYPO_SYNONYMS: Dict[str, str] = {
    "injeted": "injected",
    "injecion": "injected",
    "injec": "injected",
    "needles": "needle",
    "insuline": "insulin",
    "bandages": "bandage"
}

def is_mcq_chunk(text: str) -> bool:
    """
    Identifies multiple-choice question exam options (e.g. {'A': ..., 'B': ...}) to prevent cluttering.
    """
    if "{'A':" in text or "{'a':" in text or "Which of the following is the most appropriate" in text:
        return True
    if re.search(r"['\"]A['\"]:\s*['\"]", text) and re.search(r"['\"]B['\"]:\s*['\"]", text):
        return True
    return False

class LocalVectorDatabase:
    """
    Local TF-IDF Vector Database with exact term matching, typo normalization, and MCQ filtering.
    """
    def __init__(self, tokenizer_path: str = "tokenizer/artifacts/tokenizer.json"):
        from tokenizer.tokenizer import MedicalTokenizer
        self.tokenizer = MedicalTokenizer(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: Optional[torch.Tensor] = None
        self.idf: Optional[torch.Tensor] = None
        self.doc_frequencies: Dict[int, int] = {}

    def _tokenize_text(self, text: str) -> List[int]:
        words = re.findall(r'\w+', text.lower())
        filtered_words = []
        for w in words:
            if w in STOP_WORDS or len(w) <= 1:
                continue
            # Expand typos if matched
            expanded = TYPO_SYNONYMS.get(w, w)
            filtered_words.extend(expanded.split())
            
        token_ids = []
        for word in filtered_words:
            ids = self.tokenizer.encode(word)
            token_ids.extend([i for i in ids if i < self.vocab_size])
        return token_ids

    def _compute_tf(self, token_ids: List[int]) -> torch.Tensor:
        tf = torch.zeros(self.vocab_size)
        for tid in token_ids:
            tf[tid] += 1.0
        if len(token_ids) > 0:
            tf = tf / len(token_ids)
        return tf

    def add_chunks(self, chunks: List[Dict[str, Any]], append: bool = False):
        """
        Indexes chunks into TF-IDF vector database, filtering out MCQ choice options.
        If append is True, existing chunks are preserved and combined with new chunks.
        """
        if append and self.chunks:
            combined_chunks = self.chunks + chunks
        else:
            combined_chunks = chunks

        raw_tfs = []
        valid_chunks = []
        self.doc_frequencies = {}

        for chunk in combined_chunks:
            text = chunk["text"]
            # Skip MCQ exam choices
            if is_mcq_chunk(text):
                continue

            tids = self._tokenize_text(text)
            tf = self._compute_tf(tids)
            raw_tfs.append(tf)
            valid_chunks.append(chunk)

            unique_tids = set(tids)
            for tid in unique_tids:
                self.doc_frequencies[tid] = self.doc_frequencies.get(tid, 0) + 1

        self.chunks = valid_chunks
        num_docs = len(self.chunks)
        idf_vec = torch.zeros(self.vocab_size)
        for tid, df in self.doc_frequencies.items():
            idf_vec[tid] = math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
        self.idf = idf_vec

        tfidf_list = []
        for tf in raw_tfs:
            tfidf = tf * idf_vec
            norm = torch.norm(tfidf, p=2)
            if norm > 0:
                tfidf = tfidf / norm
            tfidf_list.append(tfidf)

        if tfidf_list:
            self.vectors = torch.stack(tfidf_list)
        else:
            self.vectors = None

    def add_document(
        self,
        text: str,
        source: str = "custom_document",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Chunks and appends a raw text document into the vector database.
        """
        from rag.chunker import MedicalChunker
        chunker = MedicalChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        new_chunks = chunker.chunk_documents([text])
        for c in new_chunks:
            c["source"] = source
        self.add_chunks(new_chunks, append=True)
        return new_chunks

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs TF-IDF Cosine Similarity Search + Typo Expansion & Keyword Boost.
        """
        if self.vectors is None or len(self.chunks) == 0 or self.idf is None:
            return []
            
        # Tokenize query with typo expansion
        q_tids = self._tokenize_text(query)
        if not q_tids:
            q_tids = [i for i in self.tokenizer.encode(query) if i < self.vocab_size]
            
        q_tf = self._compute_tf(q_tids)
        q_tfidf = q_tf * self.idf
        q_norm = torch.norm(q_tfidf, p=2)
        if q_norm > 0:
            q_tfidf = q_tfidf / q_norm
            
        q_vec = q_tfidf.unsqueeze(0)  # Shape: (1, V)
        
        # Cosine Similarity scores
        scores = F.cosine_similarity(q_vec, self.vectors, dim=1).clone()
        
        # Keyword & Typo Match Boost
        query_raw_words = set(re.findall(r'\w+', query.lower())) - STOP_WORDS
        expanded_query_words = set()
        for w in query_raw_words:
            expanded_query_words.update(TYPO_SYNONYMS.get(w, w).split())
            
        for idx, chunk in enumerate(self.chunks):
            chunk_lower = chunk["text"].lower()
            matches = sum(1 for w in expanded_query_words if w in chunk_lower)
            if matches > 0:
                scores[idx] += 1.0 * matches
                
        top_k = min(top_k, len(self.chunks))
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.chunks[idx], float(score)))
            
        return results

    def save(self, db_dir: str = "data/rag_db"):
        os.makedirs(db_dir, exist_ok=True)
        chunks_path = os.path.join(db_dir, "chunks.json")
        vec_path = os.path.join(db_dir, "vectors.pt")
        meta_path = os.path.join(db_dir, "metadata.pt")
        
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
            
        if self.vectors is not None:
            torch.save(self.vectors, vec_path)
        if self.idf is not None:
            torch.save({"idf": self.idf, "doc_frequencies": self.doc_frequencies}, meta_path)
            
        print(f"[+] Saved Vector Database ({len(self.chunks)} chunks, MCQ filtered) to: {db_dir}")

    def load(self, db_dir: str = "data/rag_db"):
        chunks_path = os.path.join(db_dir, "chunks.json")
        vec_path = os.path.join(db_dir, "vectors.pt")
        meta_path = os.path.join(db_dir, "metadata.pt")
        
        if not os.path.exists(chunks_path) or not os.path.exists(vec_path):
            raise FileNotFoundError(f"Vector Database files missing in {db_dir}")
            
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        self.vectors = torch.load(vec_path)
        if os.path.exists(meta_path):
            meta = torch.load(meta_path)
            self.idf = meta["idf"]
            self.doc_frequencies = meta["doc_frequencies"]
        print(f"[+] Loaded Vector Database ({len(self.chunks)} chunks) from: {db_dir}")
