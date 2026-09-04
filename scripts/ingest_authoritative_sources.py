#!/usr/bin/env python3
"""
Phase 13 Module: Authoritative Source RAG Ingestion Pipeline
Indexes clinical_knowledge/venipuncture_knowledge.json and vr_knowledge/venipuncture_vr_knowledge.json
into data/rag_db with full source metadata per chunk.
"""

import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase

def ingest_authoritative_rag(
    clinical_file: str = "data/clinical_knowledge/venipuncture_knowledge.json",
    vr_file: str = "data/vr_knowledge/venipuncture_vr_knowledge.json",
    db_dir: str = "data/rag_db"
):
    print("=" * 70)
    print("PHASE 13: AUTHORITATIVE RAG VECTOR DATABASE EXPANSION")
    print("=" * 70)
    
    db = LocalVectorDatabase()
    if os.path.exists(db_dir):
        try:
            db.load(db_dir)
            print(f"[+] Loaded existing vector DB from {db_dir} with {len(db.chunks):,} chunks.")
        except Exception as e:
            print(f"[!] Re-initializing database ({e}).")
            
    new_chunks = []
    
    # 1. Load Clinical Knowledge Items
    if os.path.exists(clinical_file):
        with open(clinical_file, "r", encoding="utf-8") as f:
            clin_items = json.load(f)
            
        for item in clin_items:
            chunk_text = (
                f"Topic: {item['topic']} (Step {item['step']})\n"
                f"Question: {item['question']}\n"
                f"Clinical Answer: {item['answer']}\n"
                f"Source: {item['source_id']} ({item['source_section']}, {item['source_page']})"
            )
            new_chunks.append({
                "doc_id": item["id"],
                "chunk_id": f"clin_{item['id']}",
                "text": chunk_text,
                "start_char": 0,
                "end_char": len(chunk_text),
                "source": item["source_id"],
                "topic": item["topic"],
                "step": item["step"],
                "source_section": item["source_section"],
                "source_page": item["source_page"]
            })
        print(f"[+] Formatted {len(clin_items)} authoritative clinical knowledge chunks.")

    # 2. Load VR Workflow Items
    if os.path.exists(vr_file):
        with open(vr_file, "r", encoding="utf-8") as f:
            vr_data = json.load(f)
            vr_items = vr_data.get("workflow_steps", [])
            
        for item in vr_items:
            chunk_text = (
                f"VR Workflow Step {item['step']}: {item['name']}\n"
                f"Expected Object: {item['expected_object']} | Interaction: {item['interaction']} ({item['interaction_type']})\n"
                f"VR Guidance: {item['vr_answer']}\n"
                f"Annotator: {item['annotator_guidance']}\n"
                f"Consequence: {item['invalid_interaction_consequence']}"
            )
            new_chunks.append({
                "doc_id": f"vr_step_{item['step']}",
                "chunk_id": f"vr_{item['step']}",
                "text": chunk_text,
                "start_char": 0,
                "end_char": len(chunk_text),
                "source": "SRC_VR_SIM",
                "topic": "VR Workflow Step",
                "step": item["step"],
                "source_section": f"StepManager Step {item['step']}",
                "source_page": "N/A"
            })
        print(f"[+] Formatted {len(vr_items)} VR simulation workflow chunks.")

    # 3. Add to Database & Save
    db.add_chunks(new_chunks, append=True)
    db.save(db_dir)
    
    print("=" * 70)
    print(f"[✓] RAG Vector DB successfully expanded to {len(db.chunks):,} total chunks -> {db_dir}")
    print("=" * 70)

if __name__ == "__main__":
    ingest_authoritative_rag()
