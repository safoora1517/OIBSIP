# Credit Card Fraud Detection Pipeline

A machine learning pipeline designed to detect fraudulent financial transactions under severe class imbalance using Scikit-Learn, Pandas, and Seaborn.

---

## 📌 Project Overview
Financial institutions must detect fraudulent credit card activity in real time without creating excessive false declines for genuine customers. This project analyzes a transaction corpus with a 1.51% fraud rate, applies cost-sensitive learning to counter class imbalance, benchmarks classification algorithms, and reviews production-scale streaming architectures.

---

🛠️ Tech Stack
Language: Python 3.9+

Data Processing: pandas, numpy

Machine Learning: scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler)

Visualization: matplotlib, seaborn

# Fraud Detection Machine Learning Pipeline

---

## 1. Executive Summary
This document provides full technical reporting on designing, evaluating, and stress-testing a machine learning fraud detection pipeline trained on `credit_card_fraud_10k.csv` (10,000 records).

---

## 2. Dataset Overview & Imbalance Profiling
* **Dimensions:** 10,000 transactions across 10 operational features.
* **Class Imbalance:**
  * **Legitimate (Class 0):** 9,849 (98.49%)
  * **Fraudulent (Class 1):** 151 (1.51%)
* **Missing Values:** 0 null entries across all fields.

---

## 3. Comparative Model Performance

| Metric | Logistic Regression (Cost-Sensitive) | Random Forest (Balanced Ensemble) |
| :--- | :---: | :---: |
| **Accuracy** | 95.85% | 99.15% |
| **Precision (Fraud)** | 26.55% | **100.00%** |
| **Recall (Fraud)** | **100.00%** | 43.33% |
| **F1-Score (Fraud)** | 0.4196 | **0.6047** |
| **ROC-AUC** | 0.9933 | **1.0000** |

### Trade-Off Diagnosis:
* **Logistic Regression:** Captures **100% of all fraudulent transactions** (Recall = 1.00) at the cost of flagging 83 legitimate transactions as suspicious (Precision = 26.55%).
* **Random Forest:** Achieves **100% precision** with zero false alarms, but misses 17 of the 30 fraud cases in the test split (Recall = 43.33%).
* **Deployment Recommendation:** Deploy a hybrid decision threshold or use Logistic Regression as the primary screening layer to prioritize fraud capture, followed by manual review for flagged cases.

---

## 4. Key Predictors of Fraudulent Activity
1. **`device_trust_score`:** Lower hardware integrity scores correlate heavily with unauthorized attacks.
2. **`transaction_hour`:** Fraud rates spike sharply outside normal daylight merchant operational windows.
3. **`velocity_last_24h`:** Elevated transaction frequency indicates card-testing attacks.
4. **`foreign_transaction` & `location_mismatch`:** Geography mismatches serve as strong flags for compromised credentials.