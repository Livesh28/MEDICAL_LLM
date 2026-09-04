#!/usr/bin/env python3
"""
Phases 6 & 7 Module: Knowledge Corpus & Source Version Audit
Audits data/rag_db (1,888 chunks) and data/metadata/sources.json, generating:
- outputs/rag_corpus_statistics.json
- outputs/source_audit.json
"""

import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase

def audit_corpus_and_sources():
    print("=" * 75)
    print("PHASES 6 & 7: KNOWLEDGE CORPUS & SOURCE VERSION AUDIT")
    print("=" * 75)
    
    db = LocalVectorDatabase()
    db.load("data/rag_db")
    chunks = db.chunks
    
    by_source = Counter()
    by_topic = Counter()
    by_step = Counter()
    
    empty_chunks = 0
    short_chunks = 0
    missing_source = 0
    missing_topic = 0
    missing_step = 0
    
    texts = set()
    duplicates = 0
    
    for c in chunks:
        src = c.get("source", "UNKNOWN")
        top = c.get("topic", "UNKNOWN")
        stp = c.get("step")
        txt = c.get("text", "").strip()
        
        by_source[src] += 1
        by_topic[top] += 1
        if stp is not None:
            by_step[stp] += 1
        else:
            missing_step += 1
            
        if not txt:
            empty_chunks += 1
        elif len(txt.split()) < 5:
            short_chunks += 1
            
        if src == "UNKNOWN":
            missing_source += 1
        if top == "UNKNOWN":
            missing_topic += 1
            
        if txt in texts:
            duplicates += 1
        else:
            texts.add(txt)
            
    corpus_stats = {
        "total_chunks": len(chunks),
        "unique_chunks": len(texts),
        "duplicate_chunks": duplicates,
        "empty_chunks": empty_chunks,
        "short_chunks_under_5_words": short_chunks,
        "missing_source_metadata": missing_source,
        "missing_topic_metadata": missing_topic,
        "missing_step_metadata": missing_step,
        "chunks_by_source": dict(by_source),
        "chunks_by_topic": dict(by_topic),
        "chunks_by_step": {str(k): v for k, v in sorted(by_step.items())}
    }
    
    out_corpus = "outputs/rag_corpus_statistics.json"
    with open(out_corpus, "w", encoding="utf-8") as f:
        json.dump(corpus_stats, f, indent=2)
    print(f"[+] Saved corpus statistics -> {out_corpus}")
    
    # Audit Source Registry
    sources_file = "data/metadata/sources.json"
    sources_data = []
    if os.path.exists(sources_file):
        with open(sources_file, "r", encoding="utf-8") as f:
            sources_data = json.load(f)
            
    source_audit = {
        "timestamp": "2026-09-03T10:55:00Z",
        "total_registered_sources": len(sources_data),
        "registered_sources": sources_data
    }
    
    out_sources = "outputs/source_audit.json"
    with open(out_sources, "w", encoding="utf-8") as f:
        json.dump(source_audit, f, indent=2)
    print(f"[+] Saved source audit report -> {out_sources}")
    print("=" * 75)

if __name__ == "__main__":
    audit_corpus_and_sources()
