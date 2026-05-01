# Political QA Clarity Analysis (NLP)

This repository hosts a multi-phase NLP project focused on classifying the clarity of responses in political question-answer (QA) pairs. The project is based on the **SemEval 2026 Task 6 (CLARITY)** dataset, featuring real-world political discourse.

## 🎯 Project Overview
Political figures often employ linguistic strategies to evade direct questions. This project aims to build and compare various NLP architectures to automatically detect whether a reply is:
* **Clear Reply:** Directly addresses the question.
* **Ambivalent:** Vague, diplomatic, or partially responsive.
* **Clear Non-Reply:** Completely evades the topic or refuses to answer.

---

## 🚀 Project Roadmap & Structure

The project is divided into progressive phases to demonstrate the evolution from traditional NLP techniques to state-of-the-art Deep Learning models:

### [Phase 1: ML Baselines & EDA](./01_ML_Baselines_and_EDA/)
* **Focus:** Data cleaning, Exploratory Data Analysis (EDA), and establishing initial performance baselines.
* **Techniques:** TF-IDF, Static Word Embeddings (GloVe), Logistic Regression, GridSearchCV.
* **Key Finding:** Traditional methods struggle with the contextual nuances of long, evasive political monologues.

### [Phase 2: Transformer Fine-Tuning](./02_Transformer_FineTuning/)
* **Focus:** Leveraging contextualized embeddings and attention mechanisms to capture semantic ambiguity.
* **Models:** `bert-base-uncased`, `distilbert-base-uncased`, `microsoft/deberta-v3-base`.
* **Techniques:** Custom PyTorch Training Loops, Weighted Cross-Entropy for severe class imbalance, Optuna Hyperparameter Tuning.
* **Key Finding:** **DeBERTa-v3** achieved the best generalization (Kaggle Score: 0.66) by utilizing a maximum sequence length (512) to successfully decode long-winded political "waffling".

*(Note: Future phases will be added here as the project progresses).*

---

## 🛠️ Global Tech Stack
* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch
* **NLP Libraries:** Hugging Face `transformers`, NLTK, Gensim
* **Machine Learning & Tuning:** Scikit-Learn, Optuna
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
* **Environment:** Kaggle / Jupyter Notebooks

---

## 🎓 Academic Context & Author

This project was developed as part of the **"Artificial Intelligence II: Deep Learning for Natural Language Processing"** course at the **National and Kapodistrian University of Athens (NKUA)**, Department of Informatics and Telecommunications.

* **Author:** Agapi Kallinikou
* **Academic Year:** 2025 - 2026
