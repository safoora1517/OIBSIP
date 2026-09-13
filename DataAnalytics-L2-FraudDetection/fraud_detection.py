# Library Imports & Visual Styling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set plot parameters
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

# Load Dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\OIBSIP\\DataAnalytics-L2-FraudDetection\\credit_card_fraud_10k.csv")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
df.head()

# Class Imbalance Analysis
fraud_counts = df["is_fraud"].value_counts()
fraud_percentages = (df["is_fraud"].value_counts(normalize=True) * 100).round(2)

imbalance_df = pd.DataFrame(
    {
        "Transaction Type": ["Legitimate (0)", "Fraudulent (1)"],
        "Count": [fraud_counts[0], fraud_counts[1]],
        "Percentage (%)": [fraud_percentages[0], fraud_percentages[1]],
    }
)
print(imbalance_df)

# Visualize target imbalance
plt.figure(figsize=(6, 4))
ax = sns.barplot(
    data=imbalance_df,
    x="Transaction Type",
    y="Count",
    palette=["#4C72B0", "#C44E52"],
)
plt.title("Class Distribution: Legitimate vs. Fraudulent", fontweight="bold")
plt.ylabel("Number of Transactions")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())} ({p.get_height()/len(df)*100:.2f}%)",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
    )

plt.tight_layout()
plt.show()

# Transaction Amounts & Time-of-Day Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Transaction Amount Distribution (Log scale due to right skew)
sns.boxplot(
    data=df,
    x="is_fraud",
    y="amount",
    palette=["#4C72B0", "#C44E52"],
    ax=axes[0],
)
axes[0].set_yscale("log")
axes[0].set_title(
    "Transaction Amount Distribution (Log Scale)", fontweight="bold"
)
axes[0].set_xticklabels(["Legitimate", "Fraud"])
axes[0].set_ylabel("Amount ($)")

# 2. Fraud Probability by Transaction Hour
hourly_fraud = (
    df.groupby("transaction_hour")["is_fraud"].mean().reset_index()
)
sns.lineplot(
    data=hourly_fraud,
    x="transaction_hour",
    y="is_fraud",
    marker="o",
    color="#C44E52",
    linewidth=2.5,
    ax=axes[1],
)
axes[1].set_title(
    "Fraud Rate Across Hours of the Day (0–23)", fontweight="bold"
)
axes[1].set_xlabel("Hour of Day (24-hr format)")
axes[1].set_ylabel("Fraud Rate")
axes[1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.show()

# One-Hot Encoding & Stratified Split
# Drop transaction_id as it has no predictive power
df_clean = df.drop(columns=["transaction_id"]).copy()

# One-hot encode categorical features (merchant_category)
df_encoded = pd.get_dummies(
    df_clean, columns=["merchant_category"], drop_first=True
)

X = df_encoded.drop(columns=["is_fraud"])
y = df_encoded["is_fraud"]

# Stratified 80/20 train/test split to preserve the 1.51% fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize features for numerical stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]} ({y_train.sum()} frauds)")
print(f"Testing samples:  {X_test.shape[0]} ({y_test.sum()} frauds)")

# Class Imbalance Strategy (Cost-Sensitive / SMOTE)
# Option A: class_weight='balanced' inversely weights class frequencies in the loss function:
# Weight(c) = N / (2 * N_c)
# Option B (if imbalanced-learn is installed):
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(
    "Using Cost-Sensitive Class Weighting (`class_weight='balanced'`) to penalize minority misclassifications."
)

# Train Classifiers
# Model 1: Logistic Regression (Cost-Sensitive)
lr_model = LogisticRegression(
    class_weight="balanced", random_state=42, max_iter=1000
)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Model 2: Random Forest Classifier (Balanced Subsample Ensemble)
rf_model = RandomForestClassifier(
    n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Performance Evaluation & Classification Reports
models = {
    "Logistic Regression (Balanced)": (y_pred_lr, y_prob_lr),
    "Random Forest (Balanced)": (y_pred_rf, y_prob_rf),
}

eval_rows = []
for name, (preds, probs) in models.items():
    eval_rows.append(
        {
            "Classifier": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision (Fraud)": round(
                precision_score(y_test, preds, zero_division=0), 4
            ),
            "Recall (Fraud)": round(recall_score(y_test, preds), 4),
            "F1-Score (Fraud)": round(
                f1_score(y_test, preds, zero_division=0), 4
            ),
            "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
        }
    )

print(pd.DataFrame(eval_rows))

print("\n--- Logistic Regression Classification Report ---")
print(
    classification_report(
        y_test, y_pred_lr, target_names=["Legitimate", "Fraud"], digits=4
    )
)

print("--- Random Forest Classification Report ---")
print(
    classification_report(
        y_test, y_pred_rf, target_names=["Legitimate", "Fraud"], digits=4
    )
)

# Confusion Matrix & ROC Curve Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. LR Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legit", "Fraud"],
    yticklabels=["Legit", "Fraud"],
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix: Logistic Regression", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# 2. RF Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["Legit", "Fraud"],
    yticklabels=["Legit", "Fraud"],
    ax=axes[1],
)
axes[1].set_title("Confusion Matrix: Random Forest", fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# 3. ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

axes[2].plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})",
    linewidth=2,
)
axes[2].plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})",
    linewidth=2,
)
axes[2].plot([0, 1], [0, 1], "k--", label="Random Chance")
axes[2].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate (Recall)")
axes[2].legend(loc="lower right")

plt.tight_layout()
plt.show()

# Feature Importance & Coefficient Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest Gini Importance
rf_importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values()
rf_importances.plot(kind="barh", color="#4C72B0", ax=axes[0])
axes[0].set_title("Random Forest: Feature Importance", fontweight="bold")
axes[0].set_xlabel("Relative Importance")

# Logistic Regression Feature Coefficients
lr_coefficients = pd.Series(lr_model.coef_[0], index=X.columns).sort_values()
lr_coefficients.plot(kind="barh", color="#C44E52", ax=axes[1])
axes[1].set_title(
    "Logistic Regression: Feature Coefficients (Log-Odds)", fontweight="bold"
)
axes[1].set_xlabel("Standardized Coefficient Value")
axes[1].axvline(0, color="gray", linestyle="--")

plt.tight_layout()
plt.show()