# Wine Quality Classification Pipeline

An end-to-end Machine Learning classification pipeline predicting wine quality from chemical properties using Scikit-Learn, Pandas, and Seaborn.

---

## 📌 Project Overview
Vintners require objective methods to predict wine quality prior to expert sensory evaluations. This project evaluates 1,599 red wine formulations, resolves target class imbalance through domain-justified binning, and benchmarks Random Forest, SGD, and SVC models.

---
🛠️ Tech Stack
Language: Python 3.9+

Data Processing: pandas, numpy

Machine Learning: scikit-learn (RandomForestClassifier, SGDClassifier, SVC, StandardScaler)

Visualization: matplotlib, seaborn

# Wine Quality Prediction Pipeline

---

## 1. Executive Summary
This project analyzes the physicochemical properties of 1,599 red wine samples to build a supervised classification pipeline that distinguishes good wines from average/low-quality wines.

---

## 2. Exploratory Analysis & Imbalance Resolution
* **Feature Schema:** 11 continuous input features, 0 missing records.
* **Target Distribution Anomaly:**
  * Ratings 3 and 8 account for $<2\%$ of samples.
  * Ratings 5 and 6 account for $82.5\%$ of samples.
* **Engineered Classification Boundary:**
  * Binned into binary targets: `Low/Average` ($\le 5$) vs. `Good` ($\ge 6$).
  * Achieves a balanced 46.5% / 53.5% ratio, eliminating majority-class bias without artificial sampling distortion.

---

## 3. Classifier Performance Matrix

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **80.62%** | **80.64%** | **80.51%** | **0.8057** |
| **Support Vector Classifier (SVC)** | 76.25% | 76.23% | 76.15% | 0.7618 |
| **Stochastic Gradient Descent (SGD)**| 65.62% | 67.31% | 66.27% | 0.6558 |

---

## 4. Key Chemical Quality Drivers
According to Gini Impurity reduction in the Random Forest model:
1. **Alcohol Content (17.7%):** Highest positive correlation with wine rating.
2. **Sulphates (13.7%):** Higher sulphate concentrations act as antimicrobial and antioxidant agents, improving score.
3. **Volatile Acidity (11.5%):** Strongest negative predictor; elevated acetic acid leads to vinegary defects.