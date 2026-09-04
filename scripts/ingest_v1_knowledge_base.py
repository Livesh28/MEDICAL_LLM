#!/usr/bin/env python3
"""
RAG Ingestion Pipeline V1 Script
Ingests data/rag_sources/venipuncture_rag_knowledge_base_v1.json into the existing RAG vector database
without replacing existing chunks. Rebuilds index, runs retrieval tests, verifies domain separation,
no undefined chunk IDs, and generates outputs/rag_ingestion_v1_report.json.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.database import LocalVectorDatabase
from rag.retriever_v2 import MetadataAwareHybridRetriever, tokenize_clean_text
from rag.pipeline import MedicalRAGPipeline

JSON_SOURCE_PATH = "data/rag_sources/venipuncture_rag_knowledge_base_v1.json"
DB_DIR = "data/rag_db"
REPORT_OUTPUT_PATH = "outputs/rag_ingestion_v1_report.json"

TEST_QUESTIONS = [
    "What is venipuncture?",
    "Why do we clean the site?",
    "What should I do at step 11?",
    "How does the SnapZone work?",
    "What does the controller trigger do?",
    "What should I do next?"
]

EXPECTED_DOMAINS = {
    "What is venipuncture?": ["CLINICAL"],
    "Why do we clean the site?": ["CLINICAL", "VR_GROUND_TRUTH", "VOICE_ROUTING"],
    "What should I do at step 11?": ["VR_GROUND_TRUTH"],
    "How does the SnapZone work?": ["VR_TECHNICAL"],
    "What does the controller trigger do?": ["META_QUEST", "VR_TECHNICAL"],
    "What should I do next?": ["VOICE_ROUTING", "VR_GROUND_TRUTH"]
}

def ingest_and_evaluate():
    print("=" * 70)
    print("[+] PHASE: Ingesting venipuncture_rag_knowledge_base_v1.json into RAG DB")
    print("=" * 70)

    if not os.path.exists(JSON_SOURCE_PATH):
        raise FileNotFoundError(f"Source JSON not found at: {JSON_SOURCE_PATH}")

    with open(JSON_SOURCE_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    records = kb_data.get("records", [])
    print(f"[+] Loaded {len(records)} new records from {JSON_SOURCE_PATH}")

    # 1. Load existing vector database
    db = LocalVectorDatabase()
    db.load(DB_DIR)
    existing_count = len(db.chunks)
    print(f"[+] Loaded existing vector database with {existing_count} chunks.")

    # Check existing chunk IDs to prevent duplicate chunk injection if re-run
    existing_chunk_ids = set(c.get("chunk_id", c.get("id")) for c in db.chunks)

    new_converted_chunks = []
    project_ground_truth_count = 0
    domain_counts = {}

    for record in records:
        rec_id = record["id"]
        ktype = record.get("knowledge_type", "UNKNOWN")
        topic = record.get("topic", "General")
        question = record.get("question", "")
        answer = record.get("answer", "")
        step = record.get("step", None)
        source_type = record.get("source_type", "PROJECT_GROUND_TRUTH")
        source_url = record.get("source_url", "N/A")

        # Mark project-specific records as PROJECT_GROUND_TRUTH per Requirement 6
        if ktype in ["VR_GROUND_TRUTH", "VR_TECHNICAL", "VOICE_ROUTING"]:
            source_type = "PROJECT_GROUND_TRUTH"

        if source_type == "PROJECT_GROUND_TRUTH":
            project_ground_truth_count += 1

        domain_counts[ktype] = domain_counts.get(ktype, 0) + 1

        # Format informative search text containing keywords, topic, Q&A
        extra_parts = []
        if step is not None:
            extra_parts.append(f"Step {step}")
        if "expected_interaction" in record:
            extra_parts.append(f"Interaction: {record['expected_interaction']}")
        if "intent" in record:
            extra_parts.append(f"Intent: {record['intent']}")
        
        extra_str = f" [{' | '.join(extra_parts)}]" if extra_parts else ""
        chunk_text = f"[{ktype}] Topic: {topic}. Question: {question} Answer: {answer}{extra_str}"

        chunk_dict = {
            "doc_id": f"kb_v1_{rec_id}",
            "chunk_id": rec_id,
            "id": rec_id,
            "text": chunk_text,
            "knowledge_type": ktype,
            "topic": topic,
            "step": step,
            "question": question,
            "answer": answer,
            "source_type": source_type,
            "source": source_type,
            "source_url": source_url,
            "start_char": 0,
            "end_char": len(chunk_text)
        }
        if "expected_interaction" in record:
            chunk_dict["expected_interaction"] = record["expected_interaction"]
        if "next_step" in record:
            chunk_dict["next_step"] = record["next_step"]
        if "intent" in record:
            chunk_dict["intent"] = record["intent"]

        new_converted_chunks.append(chunk_dict)

    # Filter out any duplicates if already ingested
    filtered_new_chunks = [c for c in new_converted_chunks if c["chunk_id"] not in existing_chunk_ids]
    print(f"[+] Converted {len(new_converted_chunks)} records. New unique chunks to append: {len(filtered_new_chunks)}")

    # 2. Append new chunks to existing database without replacing
    combined_chunks = db.chunks + filtered_new_chunks
    db.add_chunks(combined_chunks, append=False)
    db.save(DB_DIR)

    final_total_count = len(db.chunks)
    print(f"[+] Rebuilt RAG index. Total chunks in updated database: {final_total_count}")

    # 3. Instantiate MetadataAwareHybridRetriever with updated database
    retriever = MetadataAwareHybridRetriever(chunks=db.chunks)

    # 4. Run Retrieval Tests & Verification
    print("\n" + "=" * 70)
    print("[+] RUNNING RETRIEVAL TESTS FOR 6 TARGET QUESTIONS")
    print("=" * 70)

    retrieval_test_results = []
    all_chunks_have_valid_id = True
    domain_verification_passed = True

    for q_idx, question in enumerate(TEST_QUESTIONS):
        print(f"\n--- [Q{q_idx+1}] Question: '{question}' ---")
        
        # Step matching for step 11
        step_val = 11 if "11" in question else None
        
        scored_hits = retriever.search(query=question, top_k=5, current_step=step_val)
        
        top_5_chunks = []
        retrieved_domains = set()

        for rank, (chunk, score) in enumerate(scored_hits):
            cid = chunk.get("chunk_id", chunk.get("id"))
            if not cid or cid == "undefined":
                all_chunks_have_valid_id = False

            ktype = chunk.get("knowledge_type", "EXISTING_CORPUS")
            retrieved_domains.add(ktype)

            snippet = chunk.get("text", "")[:120] + "..."
            
            top_5_chunks.append({
                "rank": rank + 1,
                "chunk_id": cid,
                "score": round(score, 4),
                "knowledge_type": ktype,
                "topic": chunk.get("topic", "N/A"),
                "source_type": chunk.get("source_type", chunk.get("source", "N/A")),
                "snippet": snippet
            })

            print(f"  Rank {rank+1} | Score: {score:.4f} | Chunk ID: {cid} | Domain: {ktype} | Topic: {chunk.get('topic')}")

        # Domain verification
        expected = EXPECTED_DOMAINS.get(question, [])
        matched_expected = any(d in retrieved_domains for d in expected) if expected else True
        if not matched_expected:
            domain_verification_passed = False

        retrieval_test_results.append({
            "question": question,
            "expected_domains": expected,
            "retrieved_domains": list(retrieved_domains),
            "domain_match": matched_expected,
            "top_5_chunks": top_5_chunks
        })

    # 5. Build Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ingestion_summary": {
            "source_file": JSON_SOURCE_PATH,
            "initial_db_chunks": existing_count,
            "new_records_ingested": len(filtered_new_chunks),
            "final_db_chunks": final_total_count,
            "project_ground_truth_records": project_ground_truth_count,
            "domain_breakdown": domain_counts
        },
        "verifications": {
            "no_undefined_chunk_ids": all_chunks_have_valid_id,
            "old_and_new_content_coexist": (existing_count > 0 and final_total_count == existing_count + len(filtered_new_chunks)),
            "domain_relevance_verified": domain_verification_passed,
            "model_retraining_performed": False,
            "unity_vr_workflow_modified": False
        },
        "retrieval_test_results": retrieval_test_results
    }

    os.makedirs("outputs", exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[+] INGESTION & RETRIEVAL EVALUATION COMPLETED SUCCESSFULLY")
    print(f"[+] Saved report to: {REPORT_OUTPUT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    ingest_and_evaluate()
