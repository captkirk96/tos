# 📜 Terms of Service Classification using BERT

This project focuses on **classifying and analyzing Terms of Service (ToS) documents** using a fine-tuned **BERT-based NLP model**. The goal is to automatically understand, categorize, and flag important clauses in legal ToS text, enabling better transparency, compliance checks, and downstream AI applications.

---

## 🚀 Project Overview

Terms of Service documents are often long, complex, and difficult for users to understand. This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to:

* Classify ToS clauses into predefined categories
* Detect potentially **unfair, risky, or sensitive clauses**
* Enable downstream tasks like summarization, explanation, and review

The system is designed to be:

* Scalable
* Easily fine-tunable
* Integrable into larger AI agent workflows (e.g., PR review, legal analysis bots)

---

## 🎯 Use Cases

* 📌 **Clause Classification** (e.g., Data Usage, Liability, Arbitration, Termination)
* ⚠️ **Risk & Fairness Detection** (flag clauses harmful to users)
* 🧾 **Compliance & Legal Audits**
* 🤖 **AI-Assisted Legal Review Tools**
* 🔍 **Search & Retrieval over Legal Text**

---

## 🧠 Model Architecture

* **Base Model:** `bert-base-uncased`
* **Task Type:** Text / Sentence Classification
* **Training Style:** Supervised Fine-Tuning
* **Loss Function:** Cross-Entropy (single-label) or BCE (multi-label)

BERT is particularly well-suited for this task because it:

* Understands bidirectional context
* Performs well on long-form legal language
* Transfers effectively with limited labeled data

---

## 🗂️ Dataset Structure

Each ToS document is split into clauses or paragraphs and labeled.

```text
text,label
"We may share your data with third parties...",data_sharing
"This agreement is governed by the laws of...",jurisdiction
```

### Example Labels

* `data_collection`
* `data_sharing`
* `liability_limitation`
* `termination`
* `arbitration`
* `governing_law`
* `user_rights`

---

## 🏗️ Project Structure

```
terms-of-service-bert/
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels.json
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── config.yaml
├── inference/
│   └── predict.py
├── models/
│   └── bert_tos_classifier/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## 🏋️ Training Pipeline

1. **Preprocessing**

   * Clean and normalize ToS text
   * Split documents into logical clauses
   * Tokenize using BERT tokenizer

2. **Fine-Tuning**

   * Load pretrained BERT
   * Add classification head
   * Train on labeled clauses

3. **Evaluation**

   * Accuracy / F1 score
   * Class-wise performance
   * Confusion matrix analysis

4. **Model Saving**

   * Save tokenizer + model for inference

---

## 📊 Evaluation Metrics

* Accuracy
* Precision / Recall / F1-Score
* Macro & Micro averages (for class imbalance)
* Optional: Fairness-oriented metrics for risky clause detection

---

## 🔮 Inference Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

text = "We may terminate your account at any time without notice."

tokenizer = AutoTokenizer.from_pretrained("models/bert_tos_classifier")
model = AutoModelForSequenceClassification.from_pretrained("models/bert_tos_classifier")

inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
label_id = torch.argmax(outputs.logits, dim=1).item()
```

---

## ⚙️ Requirements

```text
torch
transformers
datasets
scikit-learn
numpy
pandas
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧩 Extensions & Future Work

* Multi-label classification for overlapping clauses
* Clause-level explanations using attention or SHAP
* Risk scoring and severity levels
* Integration with RAG or legal knowledge bases
* Deployment as an API or agent tool

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs for:

* New datasets
* Improved labeling schemes
* Model optimizations
* Deployment tooling

---

## 📄 License

This project is intended for **research and educational purposes**. Review and comply with legal constraints before using it in production environments.

---

## ✨ Author

Built for exploring **legal NLP and fairness-aware text classification** using modern transformer models.

