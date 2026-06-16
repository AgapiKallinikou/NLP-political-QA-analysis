
# 🧠 D3-Agentic Prompting for Political Response Classification - Overview

The proposed architecture follows an **agentic reasoning pipeline** in which multiple specialized agents cooperate to analyze a political response step-by-step before producing a final classification.

The pipeline consists of four main agents:

## 1️⃣ Question Intent Agent
Extracts the exact informational requirements of the target question.

## 2️⃣ Answer Content Agent
Identifies the relevant information actually provided in the political response.

## 3️⃣ Gap and Evasion Agent
Compares the expected information with the provided answer and detects ambiguity, evasiveness, topic shifts, or missing requirements.

## 4️⃣ Decision Agent
Produces the final response-clarity label using structured reasoning signals from the previous stages.

---

# 🖥️ Features

- Multi-agent prompting architecture
- Structured JSON intermediate reasoning
- DSPy prompt optimization
- Few-shot prompting
- Error and subgroup analysis
- Confidence-based evaluation
- Local LLM inference using Qwen-0.8B
- Explainable NLP pipeline design

---

# ⚙️ Methodology

The system decomposes the response-classification task into smaller reasoning subtasks handled by specialized agents.

Each agent produces structured intermediate outputs in JSON format, enabling:

- better interpretability,
- modular reasoning,
- easier debugging,
- and targeted prompt optimization.

DSPy was additionally used to optimize parts of the prompting pipeline under constrained Kaggle hardware resources.

---

# 📊 Key Findings

- 📌 Short and focused questions are significantly easier for the model to classify correctly.
- 📌 Long political answers dramatically reduce performance due to topic shifts and indirect rhetoric.
- 📌 Structured agent decomposition improves interpretability and error tracing.
- 📌 DSPy optimization improved the Question Intent Agent but introduced important engineering challenges under memory-constrained environments.

---

# 🛠️ Technologies Used

- Python
- DSPy
- Hugging Face Transformers
- Qwen-0.8B
- Kaggle Notebooks
- Scikit-learn
- Pandas
- NumPy

---

````
