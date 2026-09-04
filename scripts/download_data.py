#!/usr/bin/env python3
"""
Phase 2 Script: Download & Document Open Medical Datasets
Downloads open-access medical educational corpora (WikiDoc, MedQA, Medical Flashcards)
and saves raw documents along with complete provenance and license metadata.
"""

import os
import json
import argparse
from datasets import load_dataset

# Dataset metadata registry as required by project specifications
DATASET_METADATA = {
    "medical_meadow_wikidoc": {
        "hf_name": "medalpaca/medical_meadow_wikidoc",
        "description": "WikiDoc medical encyclopedia articles covering diseases, symptoms, anatomy, and treatments.",
        "license": "Apache 2.0 / CC-BY-SA",
        "source_url": "https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc",
        "intended_purpose": "Pretraining corpus for foundational medical knowledge and terminology.",
        "redistribution_permitted": True
    },
    "medical_meadow_medqa": {
        "hf_name": "medalpaca/medical_meadow_medqa",
        "description": "Medical Question-Answering pairs derived from board exam educational materials.",
        "license": "Apache 2.0 / CC-BY",
        "source_url": "https://huggingface.co/datasets/medalpaca/medical_meadow_medqa",
        "intended_purpose": "Medical reasoning, clinical inquiry, and domain-specific Q&A.",
        "redistribution_permitted": True
    },
    "medical_meadow_medical_flashcards": {
        "hf_name": "medalpaca/medical_meadow_medical_flashcards",
        "description": "Medical concepts, anatomy, physiology, and pharmacology flashcards.",
        "license": "Apache 2.0 / CC-BY",
        "source_url": "https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards",
        "intended_purpose": "High-density medical definitions and core clinical facts.",
        "redistribution_permitted": True
    }
}

def download_medical_data(raw_dir, metadata_dir, max_examples_per_ds=None):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    combined_raw_documents = []
    metadata_summary = {"datasets": {}, "total_documents": 0}
    
    print("=" * 60)
    print("PHASE 2: Downloading & Logging Open Medical Datasets")
    print("=" * 60)
    
    for key, info in DATASET_METADATA.items():
        print(f"\n[+] Processing: {info['hf_name']}...")
        try:
            split_spec = f"train[:{max_examples_per_ds}]" if max_examples_per_ds else "train"
            ds = load_dataset(info["hf_name"], split=split_spec)
            count = len(ds)
            print(f"    Loaded {count} items.")
            
            ds_docs = []
            for item in ds:
                # Format item into clean narrative text
                instruction = item.get("instruction", "").strip()
                input_text = item.get("input", "").strip()
                output_text = item.get("output", "").strip()
                
                parts = []
                if input_text:
                    parts.append(input_text)
                if output_text:
                    parts.append(output_text)
                elif instruction:
                    parts.append(instruction)
                
                doc = "\n".join(parts)
                if doc.strip():
                    ds_docs.append(doc)
            
            combined_raw_documents.extend(ds_docs)
            
            # Record metadata
            metadata_summary["datasets"][key] = {
                "hf_name": info["hf_name"],
                "description": info["description"],
                "license": info["license"],
                "source_url": info["source_url"],
                "intended_purpose": info["intended_purpose"],
                "redistribution_permitted": info["redistribution_permitted"],
                "documents_downloaded": len(ds_docs)
            }
            
        except Exception as e:
            print(f"    [!] Error downloading {info['hf_name']}: {e}")
            
    metadata_summary["total_documents"] = len(combined_raw_documents)
    
    # Save raw documents
    raw_file = os.path.join(raw_dir, "medical_corpus_raw.json")
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(combined_raw_documents, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(combined_raw_documents)} raw documents to {raw_file}")
    
    # Save metadata
    meta_file = os.path.join(metadata_dir, "dataset_info.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, indent=2)
    print(f"Saved dataset metadata provenance to {meta_file}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & Document Open Medical Corpora")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to save raw dataset")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata", help="Path to save metadata")
    parser.add_argument("--max_examples", type=int, default=5000, help="Max examples per dataset for manageable subset")
    args = parser.parse_args()
    
    download_medical_data(args.raw_dir, args.metadata_dir, args.max_examples)
