# House Price Prediction using Linear Regression

An end-to-end machine learning regression workflow predicting residential property prices using Scikit-Learn, Pandas, and Seaborn.

---

## 📌 Project Overview
Predicting property valuations requires handling continuous structural dimensions and nominal neighborhood factors. This project implements a complete linear regression pipeline—including data exploration, dummy encoding, OLS modeling, residual diagnostics, and L1/L2 regularization comparison.

---
🛠️ Tech Stack
Language: Python 3.9+

Data Processing: pandas, numpy

Machine Learning: scikit-learn (LinearRegression, Ridge, Lasso, train_test_split)

Visualization: matplotlib, seaborn

# Technical Documentation: House Price Prediction using Linear Regression

---

## 1. Executive Summary
This provides an end-to-end analytical report on building, evaluating, and interpreting an Ordinary Least Squares (OLS) Linear Regression model on the `House Price Prediction Dataset.csv` (2,000 records).

---

## 2. Feature Engineering & Preprocessing Pipeline
* **Target Feature:** `Price` (Continuous USD, mean = $537,676.86).
* **Identifier Elimination:** Dropped `Id` to eliminate non-generalizable indices.
* **Derived Attribute:** Constructed `Age` ($2026 - \text{YearBuilt}$) to express property aging while mitigating direct collinearity with the construction year.
* **Categorical Encoding:** One-Hot Encoded `Location` (`Downtown`, `Rural`, `Suburban`, `Urban`), `Condition` (`Excellent`, `Fair`, `Good`, `Poor`), and `Garage` (`Yes`, `No`) using reference class removal (`drop_first=True`).

---

## 3. Model Performance & Evaluation

| Evaluation Metric | Linear Regression (OLS) | Ridge (L2) | Lasso (L1) |
| :--- | :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | $279,859.73 | $279,859.37 | $279,859.55 |
| **Coefficient of Determination ($R^2$)** | -0.0067 | -0.0067 | -0.0067 |

### Diagnosis:
* The model's near-zero $R^2$ demonstrates that the synthetic dataset features have essentially zero linear association with the target price ($\vert{}r\vert{} < 0.06$ across all independent predictors).
* Residual analysis confirms that errors are uniformly distributed with constant variance (homoscedasticity), reflecting noise rather than systematic model bias.

---

## 4. Coefficient Interpretation
* **Top Positive Drivers:** `Floors` and `Condition_Fair` exhibit the highest positive coefficients in the fitted hyper-plane.
* **Top Negative Drivers:** `Condition_Good` and `Location_Urban` register downward shifts relative to the baseline reference classes.