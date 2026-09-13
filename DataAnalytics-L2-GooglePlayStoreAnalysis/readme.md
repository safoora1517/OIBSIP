# Google Play Store Market Analysis & NLP Pipeline

An end-to-end Exploratory Data Analysis and Sentiment Analysis project evaluating the Android app market ecosystem using Python, Pandas, VADER, Seaborn, and Plotly.

---

## 📌 Project Overview
The mobile application market is highly competitive. Success requires data-driven decision-making across category selection, pricing, app size optimization, and sentiment tracking. This project evaluates real-world Google Play Store data, engineers financial proxies, and scores user sentiment.

---

🛠️ Tech Stack
Language: Python 3.9+

Data Processing: pandas, numpy

Natural Language Processing: NLTK (VADER SentimentIntensityAnalyzer)

Visualization: matplotlib, seaborn, plotly

# Google Play Store Ecosystem Analysis

---

## 1. Executive Summary
This report analyzes 9,659 unique Android applications and companion user review sentiments from the Google Play Store to identify category saturation, pricing models, feature correlations, and consumer reception.

---

## 2. Data Cleaning & Normalization Audit
* **Row Deletion:** Removed record index `10472` due to shifted metadata fields.
* **Deduplication:** Dropped 1,181 redundant records to retain unique applications.
* **Numeric Normalization:**
  * `Installs`: Removed commas and `+` symbols, converting strings to float counts.
  * `Price`: Stripped currency symbols (`$`), creating clean numerical dollar amounts.
  * `Size`: Parsed `M` and `k` suffixes into unified Megabytes (`MB`).
* **Rating Imputation:** Filled missing rating metrics using the corresponding category median.

---

## 3. Core Findings

| Metric / Dimension | Observation | Strategic Takeaway |
| :--- | :--- | :--- |
| **App Type Distribution** | 92.2% Free vs. 7.8% Paid | Freemium architecture is mandatory for mass adoption. |
| **Top Revenue Categories** | Family, Lifestyle, Game, Finance | High user lifetime value concentrated in lifestyle and entertainment. |
| **Size vs. Installs** | Weak positive correlation ($r = 0.13$) | App scale does not guarantee viral adoption; keep lightweight. |
| **Median Paid App Price** | $2.99 (IQR: $1.49 – $4.99) | Price elasticity drops sharply above $4.99 for non-enterprise utilities. |