# Phase 3: Prompt Engineering for Political Response Clarity

This directory contains the third phase of the NLP Political Q&A analysis project. Shifting away from gradient-based fine-tuning, this phase explores **Prompt Engineering** techniques using Generative Large Language Models (Qwen3.5 family) to classify political answers into: Clear Reply, Ambivalent, or Clear Non-Reply (SemEval 2026 Task 6).

## 🎯 Objective
The goal is to design optimal prompt structures and evaluate different strategies (Zero-Shot, Few-Shot, Chain-of-Thought) without updating model weights. The challenge lies in imposing strict generation constraints to prevent base LLMs from hallucinating, rambling, or producing infinite reasoning loops.

## 📂 Folder Structure
- `code_prompting.ipynb`: Unified Kaggle inference pipeline for Qwen3.5 (0.8B, 2B, 4B) models.
- `SemEval_Clarity_Prompting_Report.pdf`: A comprehensive technical report (in Greek) detailing prompt design, hardware constraints, context truncation, and an extensive error analysis.

## 📊 Model Performance

| Model | Strategy | Macro F1 | Invalid Rate | Status / Observation |
| :--- | :--- | :---: | :---: | :--- |
| **Qwen3.5-2B** | **Zero-Shot** | **0.4320** | **0.00%** | 🏆 **Best Balance & Zero Invalid Rate** |
| Qwen3.5-2B | CoT | 0.3548 | 0.00% | Degraded rare class detection |
| Qwen3.5-4B* | Zero-Shot | 0.2720 | 4.25% | High Accuracy but biased to majority |
| Qwen3.5-0.8B | CoT | 0.2463 | 0.00% | Survived, but limited comprehension |

*Note: The 4B model required 4-bit Quantization to fit into T4 GPUs. Models were evaluated strictly using Macro F1-Score to combat class imbalance.*

## 🧠 Key Insights & Methodology

- **Generative "Leniency" vs. Discriminative "Suspicion":**
  Unlike the fine-tuned Encoders in Phase 2 which became hyper-skeptical, base Generative LLMs are inherently "lenient". They try to be helpful and often interpret diplomatic filler words as a valid `Clear Reply` (accounting for 68.9% of all errors).
- **The Accuracy Trap:**
  The 4B model achieved the highest raw Accuracy (>51%) but failed almost completely on the rarest class (`Clear Non-Reply`). It artificially inflated its score by constantly predicting the majority classes. Macro F1 was essential to expose this statistical trap.
- **Hardware Adaptations & Anchoring:**
  Strict Kaggle hardware limitations enforced 4-bit Quantization for the 4B model and Greedy Decoding (`do_sample=False`) across all models. We also used "Structural Text Anchoring" (appending `Label:` at the end of the prompt) to force immediate and valid classification.

## ⚠️ Error Analysis

Despite their massive semantic knowledge, base LLMs struggled with the mechanics of political rhetoric:
- **The Penalty of Overthinking (CoT):** Asking models to analyze the politician's strategy (Chain-of-Thought) backfired. Instead of uncovering evasion, the LLM reasoned itself into "justifying" the political dodging, crashing the `Clear Non-Reply` F1-score to just 0.12.
- **Context Window Overload (Few-Shot):** Providing 9 balanced examples (Few-Shot) bloated the context. This caused severe memory truncation in the 4B model (35% invalid rate) and cognitive overload/infinite loops in the 0.8B model (94% invalid rate).

## 🛠️ Tech Stack
- **Framework:** PyTorch (Inference)
- **Libraries:** Hugging Face `transformers`, `scikit-learn`
- **Techniques:** Prompt Engineering (Zero/Few-Shot, CoT), 4-bit Quantization, Greedy Decoding, Structural Text Anchoring.
