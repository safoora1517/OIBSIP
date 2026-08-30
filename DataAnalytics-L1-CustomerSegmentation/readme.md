# E-Commerce Customer Segmentation Analysis (RFM & K-Means)

An unsupervised Machine Learning customer segmentation pipeline utilizing Recency, Frequency, and Monetary (RFM) modeling and K-Means clustering.

---

## 📌 Project Overview
This analysis segments 350 e-commerce customers into actionable behavioral cohorts using transaction history, satisfaction ratings, and engagement metrics to guide targeted marketing and retention strategies.

---
🛠️ Tech Stack
Language: Python 3.9+

Data Processing: pandas, numpy

Machine Learning: scikit-learn (KMeans, StandardScaler)

Visualization: matplotlib, seaborn

🔍 Key Findings & Segment Profiles
VIP Champions (33.4% of users):

Average Total Spend: $1,311.14 | Items Purchased: 17.6 | Recency: 17.9 days

100% hold Gold Membership and report high customer satisfaction (mean rating: 4.68/5).

Active Value Shoppers (33.1% of users):

Average Total Spend: $629.25 | Items Purchased: 9.7 | Recency: 18.9 days

Balanced across Silver and Bronze tiers with steady order frequency.

At-Risk / Lapsed (33.4% of users):

Average Total Spend: $593.90 | Items Purchased: 10.5 | Recency: 42.9 days

Highest customer dormancy with lower average satisfaction levels.

# Technical Documentation: Customer Segmentation Analysis

---

## 1. Data Cleaning & Feature Selection
* **Hygiene:** Handled missing categorical values in `Satisfaction Level` via mode imputation.
* **Selected RFM Behavioral Attributes:**
  * **Recency:** `Days Since Last Purchase` (Mean: 26.6 days, Max: 63 days)
  * **Frequency:** `Items Purchased` (Mean: 12.6 items)
  * **Monetary:** `Total Spend` (Mean: $845.38, Max: $1,520.10)
* **Standardization:** Normalized features using `StandardScaler` ($\mu = 0, \sigma = 1$) to eliminate scale bias.

---

## 2. Clustering Methodology & K Determination
* Evaluated Within-Cluster Sum of Squares (WCSS) across $K = 1 \dots 10$.
* An elbow point was identified at **$K = 3$**, achieving clean cohort separation across monetary value and recency.

---

## 3. Targeted Marketing Strategies

1. **VIP Champions:**
   * Provide early access to new catalog releases and priority support.
   * Offer tier-maintenance rewards rather than deep price discounts.
2. **Active Value Shoppers:**
   * Introduce cross-selling bundles and spend-threshold discounts (e.g., "Spend $150 to upgrade to Gold") to increase basket size.
3. **At-Risk / Lapsed Customers:**
   * Launch automated re-engagement email campaigns with time-limited coupons.
   * Deploy customer feedback surveys to resolve friction points indicated by lower satisfaction ratings.
