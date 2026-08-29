# Retail Sales Exploratory Data Analysis (EDA)

An end-to-end Exploratory Data Analysis (EDA) on retail sales transaction data to evaluate demographic patterns, seasonal revenue trajectories, category performance, and consumer purchase behaviors.

---

## 📌 Project Overview

This repository contains an exploratory data analysis conducted on a retail sales dataset spanning 1,000 transactions recorded between **January 1, 2023** and **January 1, 2024**. The objective is to identify baseline statistical trends, time-series shifts, demographic distributions, and product performance metrics to inform commercial strategy.

---

📊 Dataset Summary
Observations: 1,000 unique transaction records

Features: 9 columns

Missing Values: 0 null entries across all fields

Total Recorded Revenue: $456,000

🛠️ Tech Stack & Dependencies
Language: Python 3.9+

Core Libraries:

pandas — Data ingestion, cleaning, manipulation, and statistical aggregation

numpy — Mathematical operations and array transformations

matplotlib — Base plotting engine and custom subplots

seaborn — Statistical visualizations, heatmaps, and styling palettes

🔍 Key FindingsRevenue Distribution: Total revenue of $456,000 is distributed evenly across categories:
Electronics: $156,905 (34.4%)
Clothing: $155,580 (34.1%)
Beauty: $143,515 (31.5%)
Customer Demographics: Customer gender is balanced (51% Female vs. 49% Male), generating comparable total revenues ($232,840 Female vs. $223,160 Male).
Correlation Matrix: Correlation between customer age, purchase quantity, and unit price is near zero ($|r| < 0.05$), indicating uniform consumption patterns across demographic cohorts.

💡 Strategic Business Recommendations
Category Bundling & Multi-Item Deals:
Because discrete unit pricing drives total order values while unit volume per transaction remains low (median: 3 units), cross-category bundles (e.g., Beauty + Clothing accessories) can lift the average transaction value.

Seasonal Inventory Management:
Align procurement and supply-chain safety stock with historical monthly order variations to prevent stockouts during peak promotional cycles.

Omni-Demographic Marketing Strategy:
Maintain balanced marketing reach across male and female customer segments across all three product lines, as transaction counts and spend remain evenly distributed.
