# Retail Sales Exploratory Data Analysis (EDA)

An end-to-end Exploratory Data Analysis (EDA) on retail sales transaction data to evaluate demographic patterns, seasonal revenue trajectories, category performance, and consumer purchase behaviors.

---

## 📌 Project Overview

This repository contains an exploratory data analysis conducted on a retail sales dataset spanning 1,000 transactions recorded between **January 1, 2023** and **January 1, 2024**. The objective is to identify baseline statistical trends, time-series shifts, demographic distributions, and product performance metrics to inform commercial strategy.

---

## 📂 Repository Structure

```text
├── data/
│   └── retail_sales_dataset.csv     # Raw transaction dataset
├── notebooks/
│   └── retail_sales_eda.ipynb       # Jupyter Notebook containing full EDA workflow
├── reports/
│   └── eda_documentation.md         # Detailed analytical documentation and observations
├── requirements.txt                 # Required Python libraries
└── README.md                        # Project summary and quickstart guide

Dataset Summary Observations: 1,000 unique transaction recordsFeatures: 9 columnsMissing Values: 0 null entries across all fieldsTotal Recorded Revenue: $456,000Feature DictionaryColumn NameData TypeDescriptionTransaction IDIntegerUnique identifier for each sales transactionDateDatetime (parsed)Date of transactionCustomer IDStringUnique customer identifierGenderStringCustomer gender (Male, Female)AgeIntegerCustomer age (range: 18–64 years)Product CategoryStringCategory of purchased merchandise (Beauty, Clothing, Electronics)QuantityIntegerUnits purchased per transaction (range: 1–4)Price per UnitIntegerUnit price in USD ($25, $30, $50, $300, $500)Total AmountIntegerTotal order revenue (Quantity $\times$ Price per Unit)🛠️ Tech Stack & DependenciesLanguage: Python 3.9+Core Libraries:pandas — Data ingestion, cleaning, manipulation, and statistical aggregationnumpy — Mathematical operations and array transformationsmatplotlib — Base plotting engine and custom subplotsseaborn — Statistical visualizations, heatmaps, and styling palettes🚀 Quickstart Guide1. Clone the repositoryBashgit clone [https://github.com/your-username/retail-sales-eda.git](https://github.com/your-username/retail-sales-eda.git)
cd retail-sales-eda
2. Create and activate a virtual environmentBash# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install required packagesBashpip install -r requirements.txt
(Contents of requirements.txt: pandas, numpy, matplotlib, seaborn, jupyter)4. Launch Jupyter NotebookBashjupyter notebook notebooks/retail_sales_eda.ipynb
🔍 Key FindingsRevenue Distribution: Total revenue of $456,000 is distributed evenly across categories:Electronics: $156,905 (34.4%)Clothing: $155,580 (34.1%)Beauty: $143,515 (31.5%)Customer Demographics: Customer gender is balanced (51% Female vs. 49% Male), generating comparable total revenues ($232,840 Female vs. $223,160 Male).Correlation Matrix: Correlation between customer age, purchase quantity, and unit price is near zero ($|r| < 0.05$), indicating uniform consumption patterns across demographic cohorts.💡 Strategic Business RecommendationsCategory Bundling & Multi-Item Deals:Because discrete unit pricing drives total order values while unit volume per transaction remains low (median: 3 units), cross-category bundles (e.g., Beauty + Clothing accessories) can lift the average transaction value.Seasonal Inventory Management:Align procurement and supply-chain safety stock with historical monthly order variations to prevent stockouts during peak promotional cycles.Omni-Demographic Marketing Strategy:Maintain balanced marketing reach across male and female customer segments across all three product lines, as transaction counts and spend remain evenly distributed.
