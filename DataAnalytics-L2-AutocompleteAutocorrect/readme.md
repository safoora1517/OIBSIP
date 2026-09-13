# Autocomplete and Autocorrect NLP Engine

A text prediction and spelling correction engine implemented in Python using Scikit-Learn, NLTK, Collections, and Seaborn.

---

## 📌 Project Overview
Autocomplete and autocorrect systems form the foundation of conversational interfaces and mobile keyboards. This repository implements frequency-weighted n-gram language models, edit-distance candidate generators, performance evaluations, and architectural comparisons against production keyboards.

---
🛠️ Tech Stack
Language: Python 3.9+

NLP & Text Processing: NLTK, collections (Counter, defaultdict), re

Evaluation & Diagnostics: scikit-learn (confusion_matrix)

Visualization: matplotlib, seaborn

# Autocomplete & Autocorrect NLP Engine

---

## 1. Executive Summary
This project implements an end-to-end NLP analytics framework evaluating statistical next-word prediction (Bigram/Trigram models) and edit-distance string correction (Levenshtein and Damerau-Levenshtein metrics).

---

## 2. Text Preprocessing & Vocabulary Architecture
* **Ingestion Pipeline:** Standardized text through lowercasing, URL/tag elimination, regex alphabetical tokenization, and optional stopword removal.
* **Lexical Coverage:** 45,900+ corpus tokens indexing over 6,700 unique vocabulary terms.

---

## 3. Autocomplete Model Performance
* **Trigram with Backoff:** Evaluates 2-word contexts $(w_{i-2}, w_{i-1})$. If the combination is unobserved, the model backs off to Bigram probabilities $P(w_i \mid w_{i-1})$, followed by Unigram priors.
* **Top-3 Prediction Coverage:** Achieves a 70% valid suggestion rate across 10 conversational prefixes.

---

## 4. Autocorrect Evaluation & Error Matrix

| Evaluation Metric | Levenshtein Candidate Model | Damerau-Levenshtein DP Model |
| :--- | :---: | :---: |
| **Correction Accuracy (20 Words)** | **90.0%** | **85.0%** |
| **Detection Precision** | 100.0% | 100.0% |
| **Detection Recall** | 90.0% | 85.0% |
| **F1-Score** | 0.947 | 0.919 |

---

## 5. Architectural Gap: Production Keyboard Comparison
* **Lack of Spatial Touch Tracking:** Our implementation treats character inputs as discrete integers rather than continuous touchscreen coordinates.
* **Absence of Long-Range Context:** Homophones (*"two"* vs. *"too"*) require contextual attention mechanisms that N-grams cannot capture beyond window length $N=3$.