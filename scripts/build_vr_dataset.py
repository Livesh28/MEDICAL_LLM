#!/usr/bin/env python3
"""
Script to build the canonical VR-Specific Knowledge Dataset data/vr_knowledge/venipuncture_vr_knowledge.json
Describing ONLY the exact 16-step VR simulation workflow, C# interactions, Annotator guidance, and StepManager rules.
"""

import os
import json

VR_STEPS = [
    {
        "step": 0,
        "name": "Wash Hands",
        "expected_object": "Sink / WaterTrigger",
        "interaction": "WaterTrigger",
        "interaction_type": "Trigger",
        "success_event": "OnTriggerEnter",
        "next_step": 1,
        "vr_answer": "Wash your hands at the Sink by placing your hands inside the WaterTrigger to perform initial hand hygiene.",
        "invalid_interaction_consequence": "Step error logged; trainee cannot wear gloves until hand hygiene is validated.",
        "annotator_guidance": "Highlights the Sink and WaterTrigger zone with blue halo."
    },
    {
        "step": 1,
        "name": "Wear Gloves",
        "expected_object": "Glove Box / Gloves",
        "interaction": "Glove Box",
        "interaction_type": "Grabbable",
        "success_event": "OnGrab",
        "next_step": 2,
        "vr_answer": "Pick up clean gloves from the Glove Box to equip personal protective equipment before touching equipment.",
        "invalid_interaction_consequence": "Touching patient or cannula without gloves triggers an IPC safety violation.",
        "annotator_guidance": "Pulsing green arrow pointing to the Glove Box on the utility table."
    },
    {
        "step": 2,
        "name": "Apply Tourniquet",
        "expected_object": "Tourniquet",
        "interaction": "Patient Upper Arm",
        "interaction_type": "Grabbable + SnapZone",
        "success_event": "OnSnap",
        "next_step": 3,
        "vr_answer": "Grab the Tourniquet and attach it to the Patient's upper arm (3-4 inches above the antecubital fossa).",
        "invalid_interaction_consequence": "Placing tourniquet directly over vein insertion site triggers placement error.",
        "annotator_guidance": "Outline highlight around upper arm tourniquet SnapZone."
    },
    {
        "step": 3,
        "name": "Take Cotton",
        "expected_object": "Cotton Box",
        "interaction": "Cotton Ball",
        "interaction_type": "Grabbable",
        "success_event": "OnGrab",
        "next_step": 4,
        "vr_answer": "Grab a clean cotton ball from the Cotton Box.",
        "invalid_interaction_consequence": "Attempting to clean arm without dipping in spirit triggers an antiseptic error.",
        "annotator_guidance": "Yellow highlight on Cotton Box."
    },
    {
        "step": 4,
        "name": "Dip Cotton",
        "expected_object": "Spirit Bottle",
        "interaction": "Spirit Bottle / LiquidTrigger",
        "interaction_type": "Trigger",
        "success_event": "OnTriggerEnter",
        "next_step": 5,
        "vr_answer": "Dip the cotton ball into the Spirit Bottle to soak it with 70% isopropyl alcohol disinfectant.",
        "invalid_interaction_consequence": "Dipping cotton in water or unapproved liquid cancels antiseptic action.",
        "annotator_guidance": "Blue indicator above Spirit Bottle opening."
    },
    {
        "step": 5,
        "name": "Clean Area",
        "expected_object": "Patient Arm",
        "interaction": "Antecubital Fossa Surface",
        "interaction_type": "Trigger",
        "success_event": "OnTriggerEnter",
        "next_step": 6,
        "vr_answer": "Rub the soaked cotton over the selected antecubital area in concentric circles for site disinfection.",
        "invalid_interaction_consequence": "Touching the site with gloved fingers after cleaning recontaminates the area and resets cleaning timer.",
        "annotator_guidance": "Circular green friction rub path on patient arm."
    },
    {
        "step": 6,
        "name": "Dispose Cotton",
        "expected_object": "Dustbin",
        "interaction": "Dustbin / WasteTrigger",
        "interaction_type": "Trigger",
        "success_event": "OnTriggerEnter",
        "next_step": 7,
        "vr_answer": "Drop the used cotton ball into the Waste Dustbin.",
        "invalid_interaction_consequence": "Leaving used cotton on sterile tray triggers biohazard waste penalty.",
        "annotator_guidance": "Red trash icon over Dustbin."
    },
    {
        "step": 7,
        "name": "Take Cannula",
        "expected_object": "Cannula",
        "interaction": "Safety Cannula / Needle Assembly",
        "interaction_type": "Grabbable",
        "success_event": "OnGrab",
        "next_step": 8,
        "vr_answer": "Pick up the sterile Safety Cannula from the equipment tray.",
        "invalid_interaction_consequence": "Uncapping needle before reaching patient arm triggers needle safety warning.",
        "annotator_guidance": "Highlight Cannula handle on tray."
    },
    {
        "step": 8,
        "name": "Insert Cannula",
        "expected_object": "Vein Trigger",
        "interaction": "Vein Trigger (Median Cubital)",
        "interaction_type": "Trigger",
        "success_event": "OnTriggerEnter",
        "next_step": 9,
        "vr_answer": "Insert the cannula into the Vein Trigger at a 15-30 degree angle with bevel facing up.",
        "invalid_interaction_consequence": "Inserting needle at >30° angle or wrong orientation triggers vein puncture fail.",
        "annotator_guidance": "3D angle guide vector showing 15-30° insertion corridor."
    },
    {
        "step": 9,
        "name": "Remove Tourniquet",
        "expected_object": "Tourniquet",
        "interaction": "Tourniquet Strap",
        "interaction_type": "Grabbable",
        "success_event": "OnUngrab",
        "next_step": 10,
        "vr_answer": "Unclip and release the Tourniquet from the patient's arm immediately after vein entry.",
        "invalid_interaction_consequence": "Leaving tourniquet tied >60s triggers hemoconcentration warning.",
        "annotator_guidance": "Flashing release icon on tourniquet buckle."
    },
    {
        "step": 10,
        "name": "Take Tube",
        "expected_object": "Blood Collection Tube",
        "interaction": "Vacutainer Tube Rack",
        "interaction_type": "Grabbable",
        "success_event": "OnGrab",
        "next_step": 11,
        "vr_answer": "Pick up the correct Blood Collection Tube from the rack following the CLSI order of draw.",
        "invalid_interaction_consequence": "Grabbing lavender tube before blue tube triggers order of draw mistake.",
        "annotator_guidance": "Glow on the next valid tube in CLSI order."
    },
    {
        "step": 11,
        "name": "Insert Tube",
        "expected_object": "Tube Slot / SnapZone",
        "interaction": "Tube Slot / SnapZone",
        "interaction_type": "SnapZone",
        "success_event": "OnSnap",
        "next_step": 12,
        "vr_answer": "Push the collection tube firmly into the Cannula Tube Slot / SnapZone until engaged.",
        "invalid_interaction_consequence": "Inserting out-of-order tube triggers SnapZone rejection and order of draw error.",
        "annotator_guidance": "SnapZone highlight on cannula holder."
    },
    {
        "step": 12,
        "name": "Blood Collection",
        "expected_object": "Blood Trigger",
        "interaction": "Blood Flow Fill Trigger",
        "interaction_type": "Trigger",
        "success_event": "OnTriggerEnter",
        "next_step": 13,
        "vr_answer": "Allow vacuum to draw blood into the tube until blood flow stops automatically.",
        "invalid_interaction_consequence": "Removing tube prematurely before vacuum fill completes causes under-filled tube error.",
        "annotator_guidance": "Red fill level bar indicator on tube."
    },
    {
        "step": 13,
        "name": "Remove Tube",
        "expected_object": "Tube",
        "interaction": "Filled Blood Tube",
        "interaction_type": "Grabbable",
        "success_event": "OnUngrab",
        "next_step": 14,
        "vr_answer": "Remove the filled blood tube from the holder and gently invert it to mix additive.",
        "invalid_interaction_consequence": "Shaking tube vigorously triggers hemolysis penalty.",
        "annotator_guidance": "Inversion animation prompt."
    },
    {
        "step": 14,
        "name": "Remove Cannula",
        "expected_object": "Cannula",
        "interaction": "Cannula Hub",
        "interaction_type": "Grabbable",
        "success_event": "OnUngrab",
        "next_step": 15,
        "vr_answer": "Withdraw the cannula swiftly from the vein and apply clean gauze pressure immediately.",
        "invalid_interaction_consequence": "Pressing gauze while needle is still inside vein causes tissue trauma error.",
        "annotator_guidance": "Smooth exit vector arrow."
    },
    {
        "step": 15,
        "name": "Dispose Cannula",
        "expected_object": "Dustbin",
        "interaction": "Biohazard Sharps Container",
        "interaction_type": "Trigger",
        "success_event": "OnTriggerEnter",
        "next_step": 16,
        "vr_answer": "Activate the needle safety guard and drop the cannula assembly into the Sharps Bin.",
        "invalid_interaction_consequence": "Disposing sharps in regular trash triggers biohazard violation.",
        "annotator_guidance": "Flashing biohazard bin indicator."
    }
]

def main():
    os.makedirs("data/vr_knowledge", exist_ok=True)
    out_file = "data/vr_knowledge/venipuncture_vr_knowledge.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"total_steps": 16, "workflow_steps": VR_STEPS}, f, indent=2)
    print(f"[+] Created VR Knowledge Dataset with {len(VR_STEPS)} workflow steps -> {out_file}")

if __name__ == "__main__":
    main()
