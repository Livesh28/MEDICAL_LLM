# Dataset Schema Specification

## 1. Clinical Knowledge Record Schema (`data/clinical_knowledge/venipuncture_knowledge.json`)

```json
{
  "id": "CLIN_000001",
  "topic": "Hand Hygiene",
  "question": "What is the primary purpose of hand hygiene prior to venipuncture?",
  "answer": "Hand hygiene eliminates transient skin microorganisms...",
  "source_id": "SRC_WHO_IPC",
  "source_section": "Section 3.1 Hand Hygiene Indications",
  "source_page": "Page 14",
  "source_url": "https://www.who.int/publications/i/item/9789241597906",
  "step": 0,
  "confidence": "high",
  "verified": true
}
```

---

## 2. VR Knowledge Record Schema (`data/vr_knowledge/venipuncture_vr_knowledge.json`)

```json
{
  "step": 11,
  "name": "Insert Tube",
  "expected_object": "Blood Collection Tube",
  "interaction": "Tube Slot / SnapZone",
  "interaction_type": "SnapZone",
  "success_event": "OnSnap",
  "next_step": 12,
  "vr_answer": "Insert the blood collection tube into the Tube Slot...",
  "invalid_interaction_consequence": "Inserting out-of-order tube triggers SnapZone rejection...",
  "annotator_guidance": "SnapZone highlight on cannula holder."
}
```

---

## 3. SFT Dataset Record Schema (`data/sft/venipuncture_sft_dataset_v3.json`)

```json
{
  "instruction": "What is the CLSI order of draw for blood collection tubes?",
  "input": "",
  "output": "The CLSI order of draw is: 1. Blood Culture -> 2. Light Blue -> 3. Red/Gold -> 4. Green -> 5. Lavender -> 6. Gray.",
  "topic": "Order of Draw",
  "step": 11,
  "source_id": "SRC_CLSI_01",
  "source_section": "Section 8.2",
  "verified": true
}
```
