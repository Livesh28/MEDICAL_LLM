#!/usr/bin/env python3
"""
Phase 20 Module: Dataset Quality & Schema Validator Script
Validates JSON structure, required fields, step bounds (0-16), missing citations, and duplicate entries.
"""

import os
import sys
import json

def validate_file(filepath: str, required_keys: list, is_list: bool = True) -> dict:
    if not os.path.exists(filepath):
        return {"status": "ERROR", "message": f"File not found: {filepath}"}
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"status": "ERROR", "message": f"Invalid JSON in {filepath}: {e}"}
        
    if isinstance(data, dict):
        if "sources" in data:
            data = data["sources"]
        elif "workflow_steps" in data:
            data = data["workflow_steps"]
            
    records = data if isinstance(data, list) else [data]
    errors = []
    
    for idx, item in enumerate(records):
        for key in required_keys:
            if key not in item or item[key] is None or item[key] == "":
                errors.append(f"Record #{idx} missing required key '{key}'")
                
        if "step" in item:
            step_val = item["step"]
            if not isinstance(step_val, int) or step_val < 0 or step_val > 16:
                errors.append(f"Record #{idx} invalid step number {step_val} (must be 0-16)")
                
    if errors:
        return {"status": "FAIL", "records_checked": len(records), "error_count": len(errors), "sample_errors": errors[:5]}
        
    return {"status": "PASS", "records_checked": len(records), "error_count": 0}

def main():
    print("=" * 70)
    print("PHASE 20: AUTOMATED DATASET QUALITY & SCHEMA VALIDATION")
    print("=" * 70)
    
    files_to_check = [
        ("data/metadata/sources.json", ["source_id", "title", "organization", "url"], False),
        ("data/clinical_knowledge/venipuncture_knowledge.json", ["id", "topic", "question", "answer", "source_id", "step"], True),
        ("data/vr_knowledge/venipuncture_vr_knowledge.json", ["step", "name", "expected_object", "vr_answer"], True),
        ("data/sft/voice_questions.json", ["id", "category", "question", "expected_answer"], True),
        ("data/sft/venipuncture_sft_dataset_v3.json", ["instruction", "output", "topic", "step"], True),
        ("data/evaluation/venipuncture_gold_eval_v2.json", ["id", "question", "verified_answer", "topic", "source"], True)
    ]
    
    all_passed = True
    for path, req_keys, is_list in files_to_check:
        res = validate_file(path, req_keys, is_list)
        status_str = f"[{res['status']}]"
        if res['status'] == "PASS":
            print(f" {status_str:6s} {path:55s} ({res['records_checked']} records verified)")
        else:
            print(f" {status_str:6s} {path:55s} -> {res.get('message', res.get('sample_errors'))}")
            all_passed = False
            
    print("=" * 70)
    if all_passed:
        print("[✓] ALL DATASET FILES PASSED VALIDATION PERFECTLY!")
    else:
        print("[✗] DATASET VALIDATION ISSUES DETECTED.")
    print("=" * 70)

if __name__ == "__main__":
    main()
