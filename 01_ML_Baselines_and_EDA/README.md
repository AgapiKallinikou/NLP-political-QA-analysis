# Phase 1: EDA & Machine Learning Baselines

This directory contains the foundational phase of the SemEval 2026 (Task 6) CLARITY project. The primary goal of this phase is to establish strong baseline performances using traditional Machine Learning techniques before advancing to complex neural networks. 

It includes extensive Exploratory Data Analysis (EDA), rigorous text preprocessing, and the evaluation of linear models using two distinct text representation methods.

## Contents
* `eda_and_baseline_models.ipynb`: The core Jupyter Notebook containing data cleaning, visual EDA (word clouds, class distribution), feature extraction, model training, and hyperparameter tuning (GridSearchCV).
* `SemEval_Clarity_Baseline_Report.pdf`: The comprehensive, 25-page academic report (in Greek) detailing the methodology, statistical analysis, and in-depth error analysis.

## Methodology & Pipeline
### 1. Text Preprocessing & EDA
Before feeding the political QA pairs into any model, rigorous text normalization was applied to handle the noisy nature of transcribed spoken language:
* **Lowercasing & Punctuation Removal:** Standardizing the text corpus.
* **Stopword Filtering:** Removing non-informative words while carefully preserving crucial semantic markers (like 'no', 'neither').
* **Tokenization & Special Tokens:** Handling specific dataset structures (e.g. using `SEPTOKEN` to distinguish between the Question and the Answer).
* **Exploratory Data Analysis:** Visualizing class distributions and generating various graphs to help identify lexical patterns across different clarity categories via data visualisation.

### 2. Modeling & Hyperparameter Tuning
To ensure a robust baseline, models were trained and tuned systematically:
* **Data Splitting:** Data was split into Training and Validation sets using the 5fold Cross Validation technique.
* **Algorithm:** Logistic Regression was chosen as the baseline classifier due to its interpretability.
* **Hyperparameter Tuning:** `GridSearchCV` was utilized with 5-fold Cross-Validation to find the optimal regularization parameters (C, penalty, solver).
* **Class Imbalance Handling:** With the *Ambivalent* class dominating (59.1%), initial brute-force training led to a high accuracy paradox but a poor Macro F1-score due to bias.  Implementing strict stratified sampling and class weight balancing was critical to force the model to recognize the minority *Clear Non-Reply* (10.3%) class. 

## Key Findings & Executive Summary

For quick reference, below is the executive summary of the findings detailed in the accompanying PDF report:

### 1. Model Performance (TF-IDF vs. Word Embeddings)
We evaluated a Logistic Regression classifier using 5-fold Cross-Validation on two different feature representations.
* **TF-IDF Vectorization:** Achieved the best performance with a **Macro F1-score of 0.5309**.
* **GloVe with Mean Pooling:** Underperformed, achieving a **Macro F1-score of 0.4281**.

**Why TF-IDF won:** In the context of political QA, specific keywords (e.g., direct affirmations/denials like *'yes'*, *'no'*, *'sorry'*) carry massive predictive weight. TF-IDF successfully isolated and heavily weighted these distinct terms. Conversely, the mean pooling approach used for GloVe embeddings smoothed out these critical signals, leading to underfitting.

### 2. Error Analysis & Limitations of Linear Baselines
While TF-IDF established a solid baseline, the error analysis reveals the fundamental limitations of purely lexical approaches:
* **Complex Evasion Tactics:** Linear models are easily fooled when a politician uses grammatically sound, relevant vocabulary to completely pivot the conversation. For example, responding with *"let's hear the second part of the question"* was misclassified because the model could not capture the semantic intent of evasion.
* **Sarcasm & Implicit Meaning:** Bag-of-Words approaches fail to capture the deep contextual syntax required to understand diplomatic pivots. 

*Conclusion:* These baseline limitations strictly validate the necessity of transitioning to sequence-to-sequence Deep Learning models and Transformer architectures (Phase 2) to capture deep semantic nuances.

---

## How to Run

1. Ensure you have the necessary libraries installed:
   ```bash
   pip install pandas numpy scikit-learn datasets matplotlib seaborn nltk gensim
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook eda_and_baseline_models.ipynb
   ```
3. Run the cells sequentially to reproduce the data cleaning, model training, and evaluation metrics.

---

## Academic Context & Author

These machine learning models were developed as part of the coursework at the **National and Kapodistrian University of Athens (NKUA)**.

* **Author:** Agapi Kallinikou
* **Academic Year:** 2025 - 2026

