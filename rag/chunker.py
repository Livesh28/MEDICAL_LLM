#!/usr/bin/env python3
"""
Phase 13 Module: Medical Document Chunker
Splits text documents into overlapping chunks for vector indexing and context retrieval.
"""

from typing import List, Dict, Any

class MedicalChunker:
    """
    Splits text documents into fixed-length text chunks with configurable overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str, doc_id: str = "doc") -> List[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        chunk_idx = 0
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end
                })
                chunk_idx += 1
                
            if end >= text_len:
                break
                
            start += (self.chunk_size - self.chunk_overlap)
            
        return chunks

    def chunk_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        all_chunks = []
        for idx, doc in enumerate(documents):
            chunks = self.chunk_document(doc, doc_id=f"doc_{idx}")
            all_chunks.extend(chunks)
        return all_chunks
