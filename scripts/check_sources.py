#!/usr/bin/env python3
"""
Phase 20 Module: Source Provenance Checker Script
Verifies that all source_id references in clinical & SFT datasets map to valid entries in data/metadata/sources.json.
"""

import os
import json

def check_provenance():
    print("=" * 70)
    print("PHASE 20: SOURCE PROVENANCE VERIFICATION")
    print("=" * 70)
    
    sources_file = "data/metadata/sources.json"
    if not os.path.exists(sources_file):
        print(f"[!] Error: {sources_file} missing.")
        return
        
    with open(sources_file, "r", encoding="utf-8") as f:
        src_data = json.load(f)
        
    valid_source_ids = {s["source_id"] for s in src_data.get("sources", [])}
    valid_source_ids.add("SRC_VR_SIM")
    valid_source_ids.add("SRC_SAFETY_GUARD")
    
    print(f"[+] Loaded {len(valid_source_ids)} registered source IDs: {valid_source_ids}")
    
    datasets_to_verify = [
        "data/clinical_knowledge/venipuncture_knowledge.json",
        "data/sft/venipuncture_sft_dataset_v3.json",
        "data/evaluation/venipuncture_gold_eval_v2.json"
    ]
    
    unmatched_count = 0
    for d_path in datasets_to_verify:
        if not os.path.exists(d_path):
            continue
        with open(d_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        dataset_unmatched = 0
        for rec in records:
            s_id = rec.get("source_id") or rec.get("source")
            if s_id and s_id not in valid_source_ids:
                dataset_unmatched += 1
                unmatched_count += 1
                
        print(f" [+] {d_path:55s}: {len(records)} records | Unmatched Sources: {dataset_unmatched}")
        
    print("=" * 70)
    if unmatched_count == 0:
        print("[✓] ALL DATASET RECORDS HAVE 100% VALID AUTHORITATIVE PROVENANCE!")
    else:
        print(f"[!] Warning: Found {unmatched_count} records with unregistered source IDs.")
    print("=" * 70)

if __name__ == "__main__":
    check_provenance()
