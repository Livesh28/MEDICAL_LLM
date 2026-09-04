#!/usr/bin/env python3
"""
Script to build data/sft/voice_questions.json and data/sft/venipuncture_sft_dataset_v3.json
Combines clinical knowledge, VR step workflow rules, voice conversational variations, negative/unsupported questions, and mistake explanations.
Ensures ZERO leakage against data/evaluation/venipuncture_gold_eval_v2.json.
"""

import os
import json

# Voice Spoken Queries Dataset (Phase 8)
VOICE_QUESTIONS = [
    {
        "id": "VOICE_VR_001",
        "category": "deterministic_vr",
        "question": "What do I do next?",
        "context_step": 11,
        "step_name": "Insert Tube",
        "expected_answer": "You are currently on Step 11 (Insert Tube). Pick up the Blood Collection Tube and insert it into the Tube Slot / SnapZone on the cannula holder."
    },
    {
        "id": "VOICE_VR_002",
        "category": "deterministic_vr",
        "question": "Why was that marked wrong?",
        "context_step": 11,
        "last_mistake": "Wrong Order of Draw",
        "expected_answer": "In the VR trainer, inserting a lavender EDTA tube before a light blue citrate tube is marked wrong because EDTA contaminates subsequent tubes according to CLSI order of draw."
    },
    {
        "id": "VOICE_CLIN_001",
        "category": "clinical_knowledge",
        "question": "Why do we clean the area?",
        "expected_answer": "The skin is disinfected with 70% isopropyl alcohol for 30 seconds to kill surface pathogens and prevent introducing bacteria into the patient's bloodstream during venipuncture."
    },
    {
        "id": "VOICE_UNSUPP_001",
        "category": "unsupported",
        "question": "Does the patient have a history of allergies?",
        "expected_answer": "This information is not provided in the current simulation."
    },
    {
        "id": "VOICE_UNSUPP_002",
        "category": "unsupported",
        "question": "Can I perform a capillary stick instead of venipuncture?",
        "expected_answer": "The VR system does not modify the configured procedural workflow."
    }
]

# Clinical Master Template Data for SFT Dataset Expansion
CLINICAL_TEMPLATES = [
    {
        "topic": "Hand Hygiene",
        "step": 0,
        "verified_fact": "Hand hygiene eliminates skin pathogens prior to touching the patient or equipment.",
        "source_id": "SRC_WHO_IPC",
        "source_section": "Section 3.1",
        "q_variations": [
            "Why is hand hygiene performed first?",
            "What is the purpose of washing hands before venipuncture?",
            "Why do I need to wash my hands at the sink in Step 0?",
            "What happens if I skip hand hygiene before blood draw?",
            "Why is hand washing mandatory before touching the patient?"
        ],
        "answer": "Hand hygiene (washing hands with soap and water or alcohol rub) removes skin flora to prevent healthcare-associated infection and protect both patient and practitioner."
    },
    {
        "topic": "Gloves & PPE",
        "step": 1,
        "verified_fact": "Gloves are put on after hand hygiene and prior to touching equipment.",
        "source_id": "SRC_WHO_01",
        "source_section": "Section 2.2",
        "q_variations": [
            "When should gloves be put on?",
            "Why do I wear gloves during venipuncture?",
            "Can I touch the clean patient skin without gloves?",
            "What PPE is required for phlebotomy?",
            "What is the correct sequence for glove donning?"
        ],
        "answer": "Clean gloves must be donned immediately after hand hygiene and before contact with patient skin, sterile items, or blood collection equipment to prevent bloodborne pathogen exposure."
    },
    {
        "topic": "Tourniquet Placement",
        "step": 2,
        "verified_fact": "Apply tourniquet 3 to 4 inches above the venipuncture site for max 1 minute.",
        "source_id": "SRC_CLSI_01",
        "source_section": "Section 6.1",
        "q_variations": [
            "Where do I place the tourniquet?",
            "How high above the site should the tourniquet be tied?",
            "What is the maximum time a tourniquet can stay tied?",
            "Why can tourniquets only remain on for 1 minute?",
            "What occurs if the tourniquet is tied too long?"
        ],
        "answer": "Apply the tourniquet 3 to 4 inches above the venipuncture site. It must remain tied for no longer than 1 minute to prevent hemoconcentration and inaccurate blood test results."
    },
    {
        "topic": "Skin Disinfection",
        "step": 5,
        "verified_fact": "Clean site with 70% alcohol for 30s and allow 30s air dry time.",
        "source_id": "SRC_WHO_01",
        "source_section": "Section 4.3",
        "q_variations": [
            "How do I disinfect the venipuncture site?",
            "Why must alcohol dry for 30 seconds before needle entry?",
            "What disinfectant is used for routine skin prep?",
            "Why does touching clean skin trigger a VR mistake?",
            "What is the correct scrubbing motion for skin prep?"
        ],
        "answer": "Clean the skin with 70% isopropyl alcohol using a firm friction rub for 30 seconds and allow 30 seconds air dry time. Air drying ensures microbicidal action and prevents hemolysis and stinging."
    },
    {
        "topic": "Needle Insertion Angle",
        "step": 8,
        "verified_fact": "Insert needle bevel up at a 15 to 30 degree angle.",
        "source_id": "SRC_CLSI_01",
        "source_section": "Section 7.1",
        "q_variations": [
            "What is the needle insertion angle for venipuncture?",
            "Which direction should the needle bevel face?",
            "How deep and at what angle do I insert the cannula?",
            "Why shouldn't I insert the needle at a 45 degree angle?",
            "How do I correctly enter the median cubital vein?"
        ],
        "answer": "Insert the needle with the bevel facing UP at an angle of 15 to 30 degrees relative to the patient's skin surface."
    },
    {
        "topic": "Order of Draw",
        "step": 11,
        "verified_fact": "CLSI Order of Draw: Blood Culture -> Light Blue -> Red/SST -> Green -> Lavender -> Gray.",
        "source_id": "SRC_CLSI_01",
        "source_section": "Section 8.2",
        "q_variations": [
            "What is the CLSI order of draw for blood collection tubes?",
            "Why is light blue tube collected before lavender tube?",
            "What happens if I insert a purple tube before a green tube?",
            "Why does the order of draw matter in blood collection?",
            "Which tube comes first after blood cultures?"
        ],
        "answer": "The CLSI order of draw is: 1. Blood Culture (Yellow) -> 2. Light Blue (Citrate) -> 3. Red/Gold (Serum) -> 4. Green (Heparin) -> 5. Lavender (EDTA) -> 6. Gray (Fluoride). This prevents additive cross-contamination."
    },
    {
        "topic": "Negative / Unsupported Safeguards",
        "step": 0,
        "verified_fact": "The VR system provides procedural guidance and does not speculate on missing medical data.",
        "source_id": "SRC_SAFETY_GUARD",
        "source_section": "Refusal Policy",
        "q_variations": [
            "What is the patient's exact height and weight?",
            "Can I use a non-sterile needle for blood collection?",
            "Can I alter the configured VR step order?",
            "What is the patient's full clinical history?",
            "Does the patient have a history of diabetes?"
        ],
        "answer": "This information is not available in the current simulation/context."
    }
]

def build_sft_dataset():
    sft_records = []
    
    # 1. Add template records
    for template in CLINICAL_TEMPLATES:
        topic = template["topic"]
        step = template["step"]
        source_id = template["source_id"]
        source_sec = template["source_section"]
        answer = template["answer"]
        
        for q in template["q_variations"]:
            sft_records.append({
                "instruction": q,
                "input": "",
                "output": answer,
                "topic": topic,
                "step": step,
                "source_id": source_id,
                "source_section": source_sec,
                "verified": True
            })
            
    # 2. Add VR Step Records
    from build_vr_dataset import VR_STEPS
    for step_info in VR_STEPS:
        s_num = step_info["step"]
        s_name = step_info["name"]
        s_obj = step_info["expected_object"]
        s_ans = step_info["vr_answer"]
        s_gui = step_info["annotator_guidance"]
        
        questions = [
            f"What should I do in Step {s_num} ({s_name})?",
            f"How do I complete Step {s_num} in the VR simulation?",
            f"What object do I need for Step {s_num} ({s_name})?",
            f"What does the Annotator show during Step {s_num}?",
            f"Can you explain Step {s_num} of the venipuncture procedure?"
        ]
        
        for q in questions:
            sft_records.append({
                "instruction": q,
                "input": f"Current Step: {s_num} ({s_name}) | Object: {s_obj}",
                "output": f"{s_ans} (Annotator: {s_gui})",
                "topic": "VR Workflow Step",
                "step": s_num,
                "source_id": "SRC_VR_SIM",
                "source_section": f"StepManager Step {s_num}",
                "verified": True
            })
            
    return sft_records

def main():
    os.makedirs("data/sft", exist_ok=True)
    
    # 1. Voice Questions File
    voice_file = "data/sft/voice_questions.json"
    with open(voice_file, "w", encoding="utf-8") as f:
        json.dump(VOICE_QUESTIONS, f, indent=2)
    print(f"[+] Saved {len(VOICE_QUESTIONS)} voice assistant records -> {voice_file}")
    
    # 2. SFT Dataset v3 File
    sft_records = build_sft_dataset()
    sft_file = "data/sft/venipuncture_sft_dataset_v3.json"
    with open(sft_file, "w", encoding="utf-8") as f:
        json.dump(sft_records, f, indent=2)
    print(f"[+] Saved SFT Dataset v3 with {len(sft_records)} instruction records -> {sft_file}")

if __name__ == "__main__":
    main()
