# Model Evaluation Benchmark & Benchmark Report

## Three-Way Comparative Evaluation Metrics

Evaluated using [`scripts/run_comparison_eval.py`](file:///Users/livesh/Medical_LLM/scripts/run_comparison_eval.py) against Gold Benchmark v2 ([`data/evaluation/venipuncture_gold_eval_v2.json`](file:///Users/livesh/Medical_LLM/data/evaluation/venipuncture_gold_eval_v2.json)):

| Model Variant | Accuracy (%) | Partial (%) | Incorrect (%) | Hallucination Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Initial 110M LM (`best.pt` - v1)** | `0.0%` | `0.0%` | `36.0%` | **`64.0%`** |
| **Improved 110M SFT (`best_v2.pt` - v2)** | `0.0%` | `16.0%` | `48.0%` | **`36.0%`** |
| **Llama 3.2 3B + Local RAG (Ollama)** | **`96.0%`** | `4.0%` | `0.0%` | **`0.0%`** |

---

## Metric Definitions
* **Accuracy:** Correct match with verified clinical facts and WHO/CLSI guidelines.
* **Partial:** Contains core clinical concept but lacks complete procedural details.
* **Incorrect:** Factually wrong clinical response.
* **Hallucination:** Unconnected, random, or invented clinical statements.
* **Refusal Quality:** Correctly handles unsupported or out-of-bound questions with safe uncertainty responses.
