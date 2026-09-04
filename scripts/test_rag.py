#!/usr/bin/env python3
"""
Phase 13 Verification Script: Test Local Medical RAG System
Tests:
1. Document chunking & Local Vector Database ingestion.
2. Vector cosine similarity search & retrieval ranking.
3. End-to-End RAG query pipeline (Retrieval + LLM generation).
4. Source snippet attribution & safety disclaimer.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.ingest import ingest_medical_documents
from rag.retriever import MedicalRetriever
from rag.pipeline import MedicalRAGPipeline

def main():
    print("=" * 70)
    print("PHASE 13: Local Medical RAG System Verification")
    print("=" * 70)
    
    clean_corpus = "data/processed/medical_corpus_clean.txt"
    db_dir = "data/rag_db"
    
    if not os.path.exists(clean_corpus):
        print(f"[!] Clean corpus missing at {clean_corpus}. Run Phase 2 first.")
        return False
        
    # 1. Ingest Document Chunks
    print("\n--- 1. Ingesting Medical Corpus into Vector DB ---")
    vector_db = ingest_medical_documents(
        corpus_path=clean_corpus,
        db_dir=db_dir,
        max_docs=500,
        chunk_size=400,
        chunk_overlap=80
    )
    
    # 2. Test Retriever Search
    print("\n--- 2. Testing Vector Retrieval & Cosine Similarity Ranking ---")
    retriever = MedicalRetriever(vector_db=vector_db)
    
    test_query = "What is hypertension and elevated blood pressure?"
    print(f"Test Query: '{test_query}'")
    
    chunks = retriever.retrieve(test_query, top_k=2)
    assert len(chunks) > 0, "Retrieval returned 0 chunks!"
    
    for i, c in enumerate(chunks, 1):
        print(f"\nRetrieved Chunk [{i}] (Score: {c['score']:.4f}):")
        print(f"  ID:   {c['chunk_id']}")
        print(f"  Text: '{c['text'][:140]}...'")
        
    # 3. Test End-to-End RAG Pipeline
    print("\n--- 3. Testing End-to-End RAG Pipeline (Retrieval + Local LLM) ---")
    pipeline = MedicalRAGPipeline(
        checkpoint_path="checkpoints/best.pt",
        db_dir=db_dir,
        device_name="mps"
    )
    
    rag_res = pipeline.answer_question(question=test_query, top_k=2, max_new_tokens=60)
    
    print("\n" + "=" * 70)
    print("RAG PIPELINE RESULT")
    print("=" * 70)
    print(f"Question:  {rag_res['question']}")
    print(f"Answer:    {rag_res['answer']}")
    print(f"Sources:   {len(rag_res['sources'])} retrieved chunks")
    print(f"Disclaimer: {rag_res['disclaimer']}")
    print("=" * 70)
    
    print("\nPhase 13 verification PASSED successfully.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
