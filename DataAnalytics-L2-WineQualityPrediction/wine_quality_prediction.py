# Environment Setup & Library Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Aesthetic settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

# Load Dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\winequality-red.csv")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
df.head()

# Data Types, Null Values & Quality Class Distribution
print("--- Data Schema & Null Count ---")
print(df.info())
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Inspect raw quality score distribution
quality_counts = df["quality"].value_counts().sort_index()
print("\nRaw Quality Score Frequencies:")
print(quality_counts)

# Visualizing target distribution
plt.figure(figsize=(8, 4))
ax = sns.barplot(
    x=quality_counts.index,
    y=quality_counts.values,
    palette="Blues_d",
    hue=quality_counts.index,
    legend=False,
)
plt.title("Distribution of Raw Wine Quality Scores (3–8)", fontweight="bold")
plt.xlabel("Quality Score")
plt.ylabel("Sample Count")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 3),
        textcoords="offset points",
    )

plt.tight_layout()
plt.show()

# Distribution of Physicochemical Features
features = [col for col in df.columns if col != "quality"]

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#3b6998", bins=25)
    axes[i].set_title(f"Distribution: {col.title()}", fontweight="bold")
    axes[i].set_xlabel("")

# Remove empty 12th subplot
fig.delaxes(axes[11])

plt.tight_layout()
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(11, 8))
corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
)
plt.title("Correlation Matrix: Chemical Properties vs. Quality", fontweight="bold")
plt.tight_layout()
plt.show()

print("--- Pearson Correlation with Wine Quality ---")
print(corr["quality"].sort_values(ascending=False).round(4))

# Target Binning
df["quality_binary"] = (df["quality"] >= 6).astype(int)

print("Engineered Binary Target Distribution:")
print(
    df["quality_binary"].value_counts(normalize=True).rename(
        {0: "0 (Low/Avg <=5)", 1: "1 (Good >=6)"}
    )
    * 100
)

# Stratified Train/Test Split & Standardization
X = df.drop(columns=["quality", "quality_binary"])
y = df["quality_binary"]

# 80/20 Stratified Split to preserve exact positive/negative class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize numerical features (required for SGD and SVC hyperplanes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples:  {X_test.shape[0]}")

# Train Random Forest, SGD, and SVC Classifiers
# Model 1: Random Forest (Tree-based ensemble; robust to unscaled inputs)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Model 2: Stochastic Gradient Descent (Linear linear model trained via SGD)
sgd_model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
sgd_model.fit(X_train_scaled, y_train)
y_pred_sgd = sgd_model.predict(X_test_scaled)

# Model 3: Support Vector Classifier (Radial Basis Function Kernel)
svc_model = SVC(kernel="rbf", C=1.0, random_state=42)
svc_model.fit(X_train_scaled, y_train)
y_pred_svc = svc_model.predict(X_test_scaled)

# Detailed Classification Reports
models = {
    "Random Forest": y_pred_rf,
    "Stochastic Gradient Descent (SGD)": y_pred_sgd,
    "Support Vector Classifier (SVC)": y_pred_svc,
}

for name, preds in models.items():
    print(f"=== {name} Classification Report ===")
    print(
        classification_report(
            y_test,
            preds,
            target_names=["Low/Avg (<=5)", "Good (>=6)"],
            digits=4,
        )
    )

# Confusion Matrix Subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = ["Low/Avg", "Good"]

for ax, (name, preds) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {name}", fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

plt.tight_layout()
plt.show()

# Random Forest Feature Importance Chart
rf_importances = pd.Series(
    rf_model.feature_importances_, index=features
).sort_values(ascending=True)

plt.figure(figsize=(9, 6))
rf_importances.plot(kind="barh", color="#2ca02c")
plt.title("Random Forest Feature Importance Analysis", fontweight="bold")
plt.xlabel("Mean Decrease in Impurity (Gini Importance)")
plt.ylabel("Physicochemical Attribute")

for idx, val in enumerate(rf_importances):
    plt.text(val + 0.003, idx, f"{val:.3f}", va="center")

plt.xlim(0, max(rf_importances) + 0.03)
plt.tight_layout()
plt.show()

# Side-by-Side Performance Comparison Table
summary_metrics = []

for name, preds in models.items():
    summary_metrics.append(
        {
            "Classifier": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision (Macro)": round(
                precision_score(y_test, preds, average="macro"), 4
            ),
            "Recall (Macro)": round(
                recall_score(y_test, preds, average="macro"), 4
            ),
            "F1-Score (Macro)": round(
                f1_score(y_test, preds, average="macro"), 4
            ),
        }
    )

comparison_df = pd.DataFrame(summary_metrics)
print(comparison_df)