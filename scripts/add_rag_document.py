#!/usr/bin/env python3
"""
Add Document to Local RAG Vector Database
Allows appending custom medical text documents or text files directly into the RAG vector index.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase

def add_document_to_rag(
    text: str = None,
    file_path: str = None,
    source_name: str = "user_added",
    db_dir: str = "data/rag_db",
    corpus_path: str = "data/processed/medical_corpus_clean.txt",
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    if not text and not file_path:
        raise ValueError("Must provide either --text or --file_path")

    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Specified file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        source_name = os.path.basename(file_path)

    text = text.strip()
    if not text:
        raise ValueError("Document text is empty.")

    print("=" * 60)
    print("ADDING DOCUMENT TO LOCAL RAG VECTOR DATABASE")
    print("=" * 60)

    db = LocalVectorDatabase()
    if os.path.exists(os.path.join(db_dir, "chunks.json")):
        db.load(db_dir)
        print(f"[+] Loaded existing vector database ({len(db.chunks):,} chunks).")
    else:
        print("[!] No existing vector database found. Creating a new database.")

    print(f"[+] Chunking & indexing document (Source: {source_name})...")
    new_chunks = db.add_document(
        text=text,
        source=source_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    db.save(db_dir)
    print(f"[+] Successfully added {len(new_chunks)} new chunks. Total RAG chunks: {len(db.chunks):,}")

    # Optionally append to raw corpus text file for model re-training
    if os.path.exists(corpus_path):
        with open(corpus_path, "a", encoding="utf-8") as f:
            f.write(f"\n<|endoftext|>\n{text}\n")
        print(f"[+] Appended document to corpus at {corpus_path}")

    print("=" * 60)
    return len(new_chunks), len(db.chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Document to RAG Database")
    parser.add_argument("--text", type=str, help="Raw document text string")
    parser.add_argument("--file_path", type=str, help="Path to text file to index")
    parser.add_argument("--source_name", type=str, default="custom_doc", help="Source identifier")
    parser.add_argument("--db_dir", type=str, default="data/rag_db", help="RAG database directory")
    args = parser.parse_args()

    add_document_to_rag(
        text=args.text,
        file_path=args.file_path,
        source_name=args.source_name,
        db_dir=args.db_dir
    )
