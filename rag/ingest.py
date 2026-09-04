#!/usr/bin/env python3
"""
Phase 13 Module: Medical RAG Data Ingestion Pipeline
Reads clean medical corpus, chunks text using MedicalChunker, and indexes into LocalVectorDatabase.
"""

import os
import argparse
from rag.chunker import MedicalChunker
from rag.database import LocalVectorDatabase

def ingest_medical_documents(
    corpus_path: str = "data/processed/medical_corpus_clean.txt",
    db_dir: str = "data/rag_db",
    max_docs: int = 1000,
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Run Phase 2 data cleaning first.")
        
    print("=" * 60)
    print("PHASE 13: Local Medical RAG Data Ingestion")
    print("=" * 60)
    print(f"Loading corpus from: {corpus_path}")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    docs = [d.strip() for d in content.split("<|endoftext|>") if d.strip()]
    if max_docs:
        docs = docs[:max_docs]
    print(f"Loaded {len(docs):,} documents for RAG vector indexing.")
    
    chunker = MedicalChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    print(f"Generated {len(chunks):,} text chunks (size={chunk_size}, overlap={chunk_overlap}).")
    
    vector_db = LocalVectorDatabase()
    print("\n[+] Embedding and indexing chunks into Local Vector Database...")
    vector_db.add_chunks(chunks)
    vector_db.save(db_dir)
    
    print("=" * 60)
    return vector_db

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Medical Documents for Local RAG")
    parser.add_argument("--corpus_path", type=str, default="data/processed/medical_corpus_clean.txt")
    parser.add_argument("--db_dir", type=str, default="data/rag_db")
    parser.add_argument("--max_docs", type=int, default=1000)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    args = parser.parse_args()
    
    ingest_medical_documents(
        corpus_path=args.corpus_path,
        db_dir=args.db_dir,
        max_docs=args.max_docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
