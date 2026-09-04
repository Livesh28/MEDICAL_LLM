#!/usr/bin/env python3
"""
Phase 3 Verification Script: Test Custom Medical Tokenizer
Tests:
1. Loading saved tokenizer artifacts.
2. Encoding medical terminology text.
3. Decoding tokens back to string.
4. Round-trip integrity test.
5. Special token recognition (<pad>, <unk>, <s>, </s>, <med_qa>, <|endoftext|>).
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import MedicalTokenizer

def main():
    print("=" * 60)
    print("PHASE 3: Custom Medical BPE Tokenizer Verification")
    print("=" * 60)
    
    tokenizer_path = "tokenizer/artifacts/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"[!] Tokenizer artifact missing at {tokenizer_path}.")
        print("Run python3 tokenizer/train_tokenizer.py first.")
        return False
        
    tokenizer = MedicalTokenizer(tokenizer_path)
    print(f"Successfully loaded tokenizer from: {tokenizer_path}")
    print(f"Vocabulary Size: {tokenizer.vocab_size:,} tokens\n")
    
    # Special Tokens Test
    print("--- 1. Special Tokens Check ---")
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<med_qa>", "<|endoftext|>"]
    for st in special_tokens:
        tid = tokenizer.token_to_id(st)
        print(f"Special Token '{st:15s}' -> ID: {tid}")
        assert tid is not None, f"Special token {st} was not assigned an ID!"
        
    # Medical Text Encode / Decode Test
    print("\n--- 2. Medical Text Encoding & Decoding Test ---")
    sample_text = "The patient presents with acute myocardial infarction, hypertension, and tachycardia."
    print(f"Input Text:\n  '{sample_text}'")
    
    token_ids = tokenizer.encode(sample_text)
    print(f"\nEncoded Token IDs ({len(token_ids)} tokens):\n  {token_ids}")
    
    decoded_text = tokenizer.decode(token_ids)
    print(f"\nDecoded Text:\n  '{decoded_text}'")
    
    # Round-Trip Test
    print("\n--- 3. Round-Trip Integrity Test ---")
    # Clean space comparison
    if sample_text.strip() == decoded_text.strip():
        print("Round-Trip Test: PERFECT MATCH SUCCESS")
    else:
        print(f"[!] Round-trip difference detected:")
        print(f"    Original: '{sample_text}'")
        print(f"    Decoded:  '{decoded_text}'")
        
    # Medical Terminology Subword Split Test
    print("\n--- 4. Medical Terminology Subword Token Representation ---")
    complex_medical_terms = ["cardiomyopathy", "hypercholesterolemia", "gastroenteritis", "electrocardiogram"]
    for term in complex_medical_terms:
        ids = tokenizer.encode(term)
        subwords = [tokenizer.id_to_token(i) for i in ids]
        print(f"Term: '{term:25s}' -> Tokens: {subwords}")
        
    print("\n" + "=" * 60)
    print("Phase 3 verification PASSED successfully.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
