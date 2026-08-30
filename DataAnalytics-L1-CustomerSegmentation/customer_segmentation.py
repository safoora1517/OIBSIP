#Environment Setup & Library Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set plot styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

# Load Dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\OIBSIP\DataAnalytics-L1-CustomerSegmentation\E-commerce Customer Behavior - Sheet1.csv")

#Data Ingestion & Data Hygiene Inspection
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("--- Column Data Types & Non-Null Values ---")
print(df.info())

print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# Handle missing values (e.g. 2 missing entries in Satisfaction Level)
df["Satisfaction Level"] = df["Satisfaction Level"].fillna(
    df["Satisfaction Level"].mode()[0]
)

print("\n--- Cleaned Data Sample ---")
df.head()

#Baseline Descriptive Statistics & Customer Metrics
# Calculate AOV, Purchase Frequency, and Total CLV Proxy
avg_purchase_value = (df["Total Spend"] / df["Items Purchased"]).mean()
avg_items_purchased = df["Items Purchased"].mean()
avg_clv = df["Total Spend"].mean()
avg_recency = df["Days Since Last Purchase"].mean()

print("--- Key Customer Baseline Metrics ---")
print(f"Average Spend per Customer (CLV Proxy): ${avg_clv:.2f}")
print(f"Average Order Value (Per Item):         ${avg_purchase_value:.2f}")
print(f"Average Purchase Volume (Items):        {avg_items_purchased:.1f} items")
print(f"Average Days Since Last Purchase:       {avg_recency:.1f} days")

# Numerical summary
df[
    [
        "Age",
        "Total Spend",
        "Items Purchased",
        "Days Since Last Purchase",
        "Average Rating",
    ]
].describe().T.round(2)

# Feature Selection & Normalization (RFM Behavioral Features)
# Recency  -> Days Since Last Purchase
# Frequency-> Items Purchased
# Monetary -> Total Spend
features = ["Days Since Last Purchase", "Items Purchased", "Total Spend"]
X = df[features].copy()

# Standardize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
X_scaled_df.head()

# Optimal Cluster Selection (Elbow Method)
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(
    k_range,
    wcss,
    marker="o",
    color="#1f77b4",
    linewidth=2.5,
    markersize=7,
)
plt.title("Elbow Method for Optimal Cluster Count (K)", fontweight="bold")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS / Inertia)")
plt.xticks(k_range)
plt.tight_layout()
plt.show()

# Apply K-Means Clustering & Profiling
optimal_k = 3
kmeans = KMeans(
    n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10
)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Segment Persona Assignment based on Cluster Characteristics
persona_map = {
    2: "VIP Champions (High Spend, High Items, Recent)",
    1: "Active Value Shoppers (Recent, Moderate Spend)",
    0: "At-Risk / Lapsed (High Inactivity, Moderate Spend)",
}
df["Customer_Segment"] = df["Cluster"].map(persona_map)

# Compute Cluster Profiling Table
cluster_summary = (
    df.groupby("Customer_Segment")
    .agg(
        {
            "Days Since Last Purchase": "mean",
            "Items Purchased": "mean",
            "Total Spend": "mean",
            "Average Rating": "mean",
            "Customer ID": "count",
        }
    )
    .rename(columns={"Customer ID": "Customer Count"})
    .round(2)
)

cluster_summary

# Cluster Scatter Plots (Multivariate Visualizations)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Recency vs. Total Spend
sns.scatterplot(
    data=df,
    x="Days Since Last Purchase",
    y="Total Spend",
    hue="Customer_Segment",
    palette="Set2",
    s=80,
    alpha=0.9,
    ax=axes[0],
)
axes[0].set_title("Recency vs. Total Spend ($)", fontweight="bold")
axes[0].set_xlabel("Days Since Last Purchase (Recency)")
axes[0].set_ylabel("Total Spend ($)")
axes[0].legend(title="Segment", loc="upper right")

# Plot 2: Frequency (Items) vs. Total Spend
sns.scatterplot(
    data=df,
    x="Items Purchased",
    y="Total Spend",
    hue="Customer_Segment",
    palette="Set2",
    s=80,
    alpha=0.9,
    ax=axes[1],
)
axes[1].set_title("Items Purchased vs. Total Spend ($)", fontweight="bold")
axes[1].set_xlabel("Items Purchased (Frequency)")
axes[1].set_ylabel("Total Spend ($)")
axes[1].legend(title="Segment", loc="upper left")

plt.tight_layout()
plt.show()

# Customer Segment Distribution Bar Chart
plt.figure(figsize=(10, 5))
seg_counts = df["Customer_Segment"].value_counts()

ax = sns.barplot(
    x=seg_counts.values, y=seg_counts.index, palette="mako", orient="h"
)
plt.title("Customer Distribution per Behavioral Segment", fontweight="bold")
plt.xlabel("Number of Customers")
plt.ylabel("Customer Segment")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_width())} ({p.get_width()/len(df)*100:.1f}%)",
        (p.get_width(), p.get_y() + p.get_height() / 2.0),
        ha="left",
        va="center",
        xytext=(5, 0),
        textcoords="offset points",
    )

plt.xlim(0, max(seg_counts.values) + 25)
plt.tight_layout()
plt.show()

# Additional Behavioral Insight (Satisfaction & Membership Correlation)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Membership Breakdown by Segment
membership_cross = pd.crosstab(
    df["Customer_Segment"], df["Membership Type"], normalize="index"
) * 100
membership_cross.plot(
    kind="barh", stacked=True, colormap="viridis", ax=axes[0]
)
axes[0].set_title("Membership Tier Distribution by Segment (%)", fontweight="bold")
axes[0].set_xlabel("Percentage (%)")
axes[0].set_ylabel("")

# Satisfaction Level by Segment
satisfaction_cross = pd.crosstab(
    df["Customer_Segment"], df["Satisfaction Level"], normalize="index"
) * 100
satisfaction_cross.plot(
    kind="barh", stacked=True, colormap="coolwarm", ax=axes[1]
)
axes[1].set_title("Satisfaction Level Distribution by Segment (%)", fontweight="bold")
axes[1].set_xlabel("Percentage (%)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
