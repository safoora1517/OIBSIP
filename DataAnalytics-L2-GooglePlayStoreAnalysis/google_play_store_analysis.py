# Library Imports & Styling
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

# Visual parameters
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["font.size"] = 11

# Download VADER lexicon for Sentiment Analysis
nltk.download("vader_lexicon", quiet=True)

# Load Primary Dataset
apps_df = pd.read_csv("C:\\Users\\HP\\Downloads\\googleplaystore.csv\\googleplaystore.csv")

# Load Companion Review Dataset (handles existence or graceful fallback)
try:
    reviews_df = pd.read_csv("C:\\Users\\HP\\Downloads\\googleplaystore_user_reviews.csv")
    print(
        f"Reviews Dataset Loaded: {reviews_df.shape[0]} rows, {reviews_df.shape[1]} columns"
    )
except FileNotFoundError:
    reviews_df = None
    print(
        "User reviews file not found locally. A synthetic/cached schema will be initialized."
    )

print(f"Apps Dataset Loaded: {apps_df.shape[0]} rows, {apps_df.shape[1]} columns")
apps_df.head(3)

# Data Cleaning Pipeline
df = apps_df.copy()

# 1. Remove corrupted row (Row 10472 where Category is shifted to '1.9')
if 10472 in df.index and df.loc[10472, "Category"] == "1.9":
    df.drop(index=10472, inplace=True)

# 2. Remove duplicate app entries
initial_count = len(df)
df.drop_duplicates(subset=["App"], inplace=True)
print(f"Removed {initial_count - len(df)} duplicate app records.")

# 3. Clean 'Installs' (e.g. '10,000+' -> 10000.0)
df["Installs_Numeric"] = (
    df["Installs"]
    .str.replace("+", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# 4. Clean 'Price' (e.g. '$4.99' -> 4.99)
df["Price_Numeric"] = (
    df["Price"].str.replace("$", "", regex=False).astype(float)
)

# 5. Clean 'Reviews'
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

# 6. Parse 'Size' into MB (e.g. '19M' -> 19.0, '512k' -> 0.5, 'Varies with device' -> NaN)
def parse_size(size_str):
    if isinstance(size_str, str):
        if "M" in size_str:
            return float(size_str.replace("M", ""))
        elif "k" in size_str:
            return float(size_str.replace("k", "")) / 1024.0
    return np.nan


df["Size_MB"] = df["Size"].apply(parse_size)

# 7. Impute missing Ratings with category median
df["Rating"] = df.groupby("Category")["Rating"].transform(
    lambda x: x.fillna(x.median())
)

# 8. Calculate Estimated Revenue proxy (Installs * Price)
df["Estimated_Revenue"] = df["Installs_Numeric"] * df["Price_Numeric"]

# 9. Clean 'Type'
df["Type"] = df["Type"].fillna("Free")
df = df[df["Type"].isin(["Free", "Paid"])].copy()

print("\n--- Cleaned Dataset Overview ---")
print(
    df[
        [
            "App",
            "Category",
            "Rating",
            "Reviews",
            "Size_MB",
            "Installs_Numeric",
            "Price_Numeric",
            "Estimated_Revenue",
        ]
    ].info()
)

# App Distribution Across Categories
cat_counts = df["Category"].value_counts()

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x=cat_counts.values, y=cat_counts.index, palette="viridis", orient="h"
)
plt.title(
    "App Saturation: Total Number of Apps per Category", fontweight="bold"
)
plt.xlabel("Number of Unique Apps")
plt.ylabel("Category")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_width())}",
        (p.get_width(), p.get_y() + p.get_height() / 2.0),
        ha="left",
        va="center",
        xytext=(4, 0),
        textcoords="offset points",
    )

plt.tight_layout()
plt.show()

print("Top 5 Most Saturated Categories:")
print(cat_counts.head(5))

# Ratings Distribution and Category Averages
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Overall Rating Distribution
sns.histplot(df["Rating"], bins=25, kde=True, color="#2b5c8f", ax=axes[0])
axes[0].axvline(
    df["Rating"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {df['Rating'].mean():.2f}",
)
axes[0].axvline(
    df["Rating"].median(),
    color="orange",
    linestyle="-",
    label=f"Median: {df['Rating'].median():.2f}",
)
axes[0].set_title("Distribution of App Ratings", fontweight="bold")
axes[0].set_xlabel("Rating (1.0 to 5.0)")
axes[0].legend()

# 2. Top 10 Highest Rated Categories
top_rated_cats = (
    df.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
sns.barplot(
    x=top_rated_cats.values,
    y=top_rated_cats.index,
    palette="mako",
    ax=axes[1],
    orient="h",
)
axes[1].set_xlim(3.8, 4.6)
axes[1].set_title("Top 10 Categories by Average Rating", fontweight="bold")
axes[1].set_xlabel("Average Rating")

plt.tight_layout()
plt.show()

# Scatter Plot: Size vs. Installs
plt.figure(figsize=(10, 6))
clean_size_df = df.dropna(subset=["Size_MB"]).copy()

sns.scatterplot(
    data=clean_size_df,
    x="Size_MB",
    y="Installs_Numeric",
    alpha=0.4,
    color="#2b5c8f",
    s=40,
)
plt.yscale("log")
plt.title("App Size (MB) vs. Number of Installs (Log Scale)", fontweight="bold")
plt.xlabel("Size (MB)")
plt.ylabel("Installs (Log Scale)")

# Calculate Correlation
corr_val = clean_size_df[["Size_MB", "Installs_Numeric"]].corr().iloc[0, 1]
plt.annotate(
    f"Pearson Correlation: r = {corr_val:.2f}",
    xy=(0.05, 0.90),
    xycoords="axes fraction",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
)

plt.tight_layout()
plt.show()

# Paid vs. Free & Category Revenue Potential
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Paid vs. Free Distribution
type_counts = df["Type"].value_counts()
axes[0].pie(
    type_counts,
    labels=type_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=["#5b9bd5", "#ed7d31"],
    explode=(0, 0.08),
)
axes[0].set_title("Free vs. Paid App Breakdown", fontweight="bold")

# 2. Price Distribution for Paid Apps (Filtering anomalous '$400' joke apps)
paid_apps = df[(df["Type"] == "Paid") & (df["Price_Numeric"] < 50)].copy()
sns.histplot(
    paid_apps["Price_Numeric"], bins=20, kde=True, color="#ed7d31", ax=axes[1]
)
axes[1].set_title("Price Distribution for Paid Apps (<$50)", fontweight="bold")
axes[1].set_xlabel("Price ($)")

# 3. Top 5 Categories by Estimated Revenue
top_rev = (
    df.groupby("Category")["Estimated_Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
sns.barplot(
    x=top_rev.values / 1e6,
    y=top_rev.index,
    palette="rocket",
    ax=axes[2],
    orient="h",
)
axes[2].set_title("Top 5 Categories by Est. Revenue ($M)", fontweight="bold")
axes[2].set_xlabel("Revenue ($ Millions)")

plt.tight_layout()
plt.show()

# VADER Sentiment Scoring on Reviews
sia = SentimentIntensityAnalyzer()

if reviews_df is not None:
    rev_clean = reviews_df.dropna(subset=["Translated_Review"]).copy()
    rev_clean["Compound_Score"] = rev_clean["Translated_Review"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
else:
    # Synthetic/demonstration review sample if companion file is not provided
    sample_reviews = [
        ("Photo Editor", "Love this app, very intuitive and powerful!", "ART_AND_DESIGN"),
        ("Photo Editor", "Crashes constantly after the latest update.", "ART_AND_DESIGN"),
        ("Coloring book", "Fun and relaxing, my kids enjoy it.", "ART_AND_DESIGN"),
        ("Clash of Clans", "Addictive gameplay but high microtransactions.", "GAME"),
        ("Candy Crush", "Amazing game, best puzzle experience ever.", "GAME"),
        ("Finance Tracker", "Useful tool, makes budget management effortless.", "FINANCE"),
    ]
    rev_clean = pd.DataFrame(
        sample_reviews, columns=["App", "Translated_Review", "Category"]
    )
    rev_clean["Compound_Score"] = rev_clean["Translated_Review"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

# Classify polarity
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


rev_clean["Sentiment_Label"] = rev_clean["Compound_Score"].apply(
    classify_sentiment
)

# Merge category from apps dataset if missing in reviews
if "Category" not in rev_clean.columns:
    rev_clean = rev_clean.merge(
        df[["App", "Category"]], on="App", how="inner"
    )

# Visualize Sentiment Breakdown
plt.figure(figsize=(7, 4))
sns.countplot(
    data=rev_clean,
    x="Sentiment_Label",
    palette=["#55A868", "#C44E52", "#4C72B0"],
    order=["Positive", "Neutral", "Negative"],
)
plt.title("User Review Sentiment Distribution", fontweight="bold")
plt.xlabel("Sentiment Class")
plt.ylabel("Review Count")
plt.tight_layout()
plt.show()

# Category Sentiment Comparison
cat_sentiment = (
    rev_clean.groupby("Category")["Sentiment_Label"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)
if "Positive" in cat_sentiment.columns and "Negative" in cat_sentiment.columns:
    cat_sentiment["Pos_to_Neg_Ratio"] = (
        cat_sentiment["Positive"] / (cat_sentiment["Negative"] + 1e-5)
    )
    top_pos_cats = cat_sentiment.sort_values(
        by="Positive", ascending=False
    ).head(8)

    top_pos_cats[["Positive", "Neutral", "Negative"]].plot(
        kind="barh",
        stacked=True,
        figsize=(10, 6),
        color=["#55A868", "#4C72B0", "#C44E52"],
    )
    plt.title("Sentiment Proportions Across Top Categories", fontweight="bold")
    plt.xlabel("Proportion")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # Interactive Bubble Plot: Installs vs. Rating vs. Category
fig_interactive = px.scatter(
    df[df["Reviews"] > 5000].dropna(subset=["Rating"]),
    x="Size_MB",
    y="Rating",
    size="Installs_Numeric",
    color="Category",
    hover_name="App",
    log_x=True,
    size_max=45,
    title="Interactive Ecosystem: App Size vs. Rating (Size = Installs)",
    labels={"Size_MB": "App Size (MB, Log Scale)", "Rating": "Average Rating"},
    template="plotly_white",
)
fig_interactive.show()