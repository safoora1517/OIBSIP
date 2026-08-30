# NFL Play-by-Play Data Cleaning Pipeline

A structured data cleaning and transformation pipeline using Python and pandas to convert messy sports play-by-play data into an analysis-ready dataset.

---

## 📌 Project Overview
Sports play-by-play data contains extensive missing values due to conditional events (e.g., kickoffs having no downs, timeouts having no team possession). This project implements an audit-driven cleaning workflow that isolates valid plays, imputes missing metrics using domain logic, normalizes text columns, and validates outliers.

---
🛠️ Tech Stack
Language: Python 3.9+

Data Manipulation & Cleaning: pandas, numpy

Exploratory Visualizations: matplotlib, seaborn