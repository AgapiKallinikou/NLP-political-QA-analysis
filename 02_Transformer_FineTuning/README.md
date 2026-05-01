# Phase 2: Transformer Fine-Tuning for Political Response Clarity

This directory contains the second phase of the NLP Political Q&A analysis project. Moving beyond traditional machine learning baselines, this phase leverages state-of-the-art Transformer architectures to classify political answers into three categories: **Clear Reply**, **Ambivalent**, or **Clear Non-Reply** (SemEval 2026 Task 6).

## 🎯 Objective
The goal is to develop an effective text classifier by fine-tuning pretrained language models on the CLARITY dataset. To maintain granular control over the optimization process, custom PyTorch training loops were implemented from scratch, bypassing the standard Hugging Face Trainer API.

## 📂 Folder Structure
* `microsoft-deberta-v3-base.ipynb`: Custom training & inference for **DeBERTa-v3** (Best Performing Model).
* `bert-base-uncased.ipynb`: Custom training & inference for **BERT-base**.
* `distilbert-base-uncased.ipynb`: Custom training & inference for **DistilBERT**.
* `SemEval_Clarity_Transformers_Report.pdf`: A comprehensive 27-page technical report (in Greek) detailing data formulation, hyperparameter tuning, learning/ROC curves & confusion matrixes, and an extensive error analysis.

## 📊 Model Performance

| Model | Val Accuracy | Val Macro F1 | Kaggle Score (Test) | Status / Observation |
| :--- | :---: | :---: | :---: | :--- |
| **DeBERTa-v3** | 0.6366 | 0.5953 | **0.6600** | 🏆 **Best Generalization** |
| **BERT-base** | 0.6411 | 0.6102 | 0.6400 | Slight Overfitting |
| **DistilBERT** | 0.5676 | 0.5570 | 0.5800 | Underfitting (Limited Capacity) |

*Note: Models were evaluated strictly using **Macro F1-Score** due to severe class imbalance (the 'Ambivalent' class heavily dominates political discourse).*

## 🧠 Key Insights & Methodology

1. **The "Waffling" Factor (Sequence Length):**
   Politicians frequently use long, diplomatic monologues to stall or frame their answers. Utilizing the maximum sequence length (`max_length = 512`) was crucial. Smaller context windows caused models to miss the actual answer hidden deep within the text, defaulting to 'Ambivalent'.
2. **Handling Class Imbalance (Loss Weights):**
   Inverse class weights were injected directly into the PyTorch `CrossEntropyLoss` (e.g., 9.69 for Clear Non-Reply). This successfully forced the models to identify minority classes, albeit making them highly "skeptical" and strict when identifying standard Clear Replies.
3. **Softmax Decision Boundaries:**
   Analysis of the softmax probabilities revealed that the models' errors were not random. In misclassified "Ambivalent" cases, the models were often split nearly 50-50, perfectly reflecting the inherent semantic ambiguity of the human political language.

## ⚠️ Error Analysis
Despite high performance, Transformers struggled with specific linguistic patterns:
* **Misleading Introductions:** A politician starting with "Well, no, Major..." tricks the model into an immediate 'Clear Non-Reply' prediction, ignoring the massive ambivalent evasion that follows.
* **Defensive Posturing:** Phrases like *"I will be rendering no opinion from the podium"* cause models to hyper-focus on the negation, confusing diplomatic ambivalence with a hard refusal to answer.

## 🛠️ Tech Stack
* **Framework:** PyTorch
* **Libraries:** Hugging Face `transformers`, `Optuna` (for hyperparameter tuning), `scikit-learn`.
* **Techniques:** Dynamic Contextualized Embeddings, Subword Tokenization, Custom Training Loops, Stratified Splitting, Weighted Cross-Entropy.
