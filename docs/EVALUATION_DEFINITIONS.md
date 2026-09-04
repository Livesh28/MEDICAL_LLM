# Official Evaluation Criteria & Metric Definitions

## Overview
This document defines the strict, reproducible evaluation criteria used across all comparative benchmarks for the **Medical VR – Venipuncture Training Simulation**.

---

## 1. Metric Classification Criteria

### A. Correct
* **Definition:** The generated response materially matches the verified clinical ground truth, incorporates essential safety/procedural points, and accurately answers the question.
* **Score Impact:** Counted in primary Accuracy percentage.

### B. Partial (Partially Correct)
* **Definition:** The core clinical principle stated by the model is factually correct, but minor procedural details (such as exact inversion counts or dry time seconds) are missing or phrased loosely.
* **Score Impact:** Tracked separately in Partial percentage. Combined with Correct for Total Clinical Guidance Coverage.

### C. Incorrect
* **Definition:** The response contains factual errors, misidentifies equipment, or contradicts WHO/CLSI guidelines (e.g. recommending tourniquet placement *below* the vein).
* **Score Impact:** Counted as a procedural failure.

### D. Hallucination
* **Definition:** The response introduces invented medical claims, unverified clinical steps, or non-existent patient conditions not supported by the retrieved evidence corpus.
* **Score Impact:** Counted as a severe safety failure.

### E. Unsupported (Correct Refusal)
* **Definition:** The question requests out-of-scope patient information (e.g., blood pressure, medical history, prescribing changes). The system accurately recognizes that evidence is lacking and issues a safe refusal (*"This information is not provided in the current simulation."*).
* **Score Impact:** Counted as a Correct Refusal.

---

## 2. Combined Accuracy Formula
$$\text{Primary Accuracy (\%)} = \frac{\text{Correct} + \text{Correct Refusal}}{\text{Total Questions}} \times 100$$
$$\text{Total Guidance Coverage (\%)} = \frac{\text{Correct} + \text{Correct Refusal} + \text{Partial}}{\text{Total Questions}} \times 100$$
