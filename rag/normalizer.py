#!/usr/bin/env python3
"""
Phase 3 Module: Controlled Venipuncture Query Normalizer
Normalizes raw user queries with controlled medical abbreviation mapping and venipuncture concept expansion.
Does NOT change the core meaning of the query.
"""

import re
from typing import Dict

# Controlled Venipuncture Abbreviation & Concept Dictionary
VENIPUNCTURE_ABBREVIATIONS: Dict[str, str] = {
    r"\bppe\b": "personal protective equipment gloves",
    r"\bsop\b": "standard operating procedure protocol",
    r"\bclsi\b": "Clinical Laboratory Standards Institute GP41",
    r"\bwho\b": "World Health Organization phlebotomy guidance",
    r"\bedta\b": "lavender top EDTA tube anticoagulant",
    r"\bsst\b": "gold top serum separator tube",
}

VENIPUNCTURE_CONCEPT_MAP: Dict[str, str] = {
    r"\bcleaning\b": "skin preparation disinfected alcohol",
    r"\bdisinfect(ion|ing|ed)?\b": "skin preparation disinfected 70 percent isopropyl alcohol",
    r"\bcollection tube(s)?\b": "blood collection tube order of draw vacutainer",
    r"\bneedle(s)?\b": "cannula needle safety assembly",
    r"\btourniquet(s)?\b": "tourniquet application upper arm antecubital fossa",
}

def normalize_query(query: str) -> str:
    """
    Normalizes input query for medical RAG retrieval while keeping original text intact.
    """
    q_norm = query.lower().strip()
    
    # 0. Separate joined step numbers (e.g. "6step" -> "step 6", "step6" -> "step 6")
    q_norm = re.sub(r"\b(\d+)\s*step(s)?\b", r"step \1", q_norm)
    q_norm = re.sub(r"\bstep\s*(\d+)\b", r"step \1", q_norm)

    # 1. Expand abbreviations
    for pattern, expansion in VENIPUNCTURE_ABBREVIATIONS.items():
        q_norm = re.sub(pattern, expansion, q_norm)
        
    # 2. Expand venipuncture concepts
    for pattern, expansion in VENIPUNCTURE_CONCEPT_MAP.items():
        q_norm = re.sub(pattern, expansion, q_norm)
        
    # 3. Clean extra whitespace
    q_norm = re.sub(r"\s+", " ", q_norm).strip()
    return q_norm
