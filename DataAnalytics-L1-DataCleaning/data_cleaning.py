# Library Imports & Setup
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visual parameters
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

# Load dataset (or sample if working on memory-constrained systems)
# Note: Low_memory=False handles DtypeWarnings on large play-by-play files
csv_path = os.path.join(
    "C:\\", "Users", "HP", "OneDrive", "Desktop", "OIBSIP",
    "DataAnalytics-L1-_DataCleaning", "NFL Play by Play 2009-2016 (v3).csv"
)
raw_df = pd.read_csv(csv_path, low_memory=False)

print(f"Dataset Raw Dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
raw_df.head()

# Data Quality Assessment Report
print("=== DATA QUALITY ASSESSMENT REPORT ===")

# 1. Total & Percentage Missing per Column
null_counts = raw_df.isnull().sum()
null_percent = (raw_df.isnull().sum() / len(raw_df)) * 100
dq_report = pd.DataFrame(
    {
        "Data_Type": raw_df.dtypes,
        "Null_Count": null_counts,
        "Null_Percentage": null_percent.round(2),
        "Unique_Values": raw_df.nunique(),
    }
)

# Filter to columns containing missing data
columns_with_nulls = dq_report[dq_report["Null_Count"] > 0].sort_values(
    by="Null_Count", ascending=False
)
print(f"Total Columns with Missing Values: {len(columns_with_nulls)} / {raw_df.shape[1]}")
print(columns_with_nulls.head(20))

# 2. Duplicate Check
duplicate_rows = raw_df.duplicated().sum()
print(f"\nExact Duplicate Rows Detected: {duplicate_rows}")

# Feature Pruning & Row Filtering
df = raw_df.copy()

# Step A: Drop ultra-sparse columns (>90% missing values)
# These represent rare events (e.g., TwoPointConv, DefTwoPoint) better represented as indicators
sparse_threshold = 0.90
sparse_cols = df.columns[df.isnull().mean() > sparse_threshold].tolist()
print(f"Dropping {len(sparse_cols)} columns with >90% missing values:")
print(sparse_cols[:10], "...")

df.drop(columns=sparse_cols, inplace=True)

# Step B: Filter out non-play events (Quarter End, Two Minute Warning, Timeouts)
# where key attributes like posteam, down, and yards gained are non-applicable
valid_plays = df[
    df["PlayType"].notnull()
    & (~df["PlayType"].isin(["Quarter End", "Two Minute Warning", "Timeout"]))
].copy()
print(f"Retained active play records: {valid_plays.shape[0]} rows")

# Imputation Logic per Column Type
# 1. Down & Distance: Impute missing downs with forward-fill within the same game drive
valid_plays["down"] = (
    valid_plays.groupby(["GameID", "Drive"])["down"]
    .ffill()
    .bfill()
    .fillna(1)
)

# 2. Field Location (yrdline100, yrdln): Impute using median field position
valid_plays["yrdline100"] = valid_plays["yrdline100"].fillna(
    valid_plays["yrdline100"].median()
)
valid_plays["yrdln"] = valid_plays["yrdln"].fillna(valid_plays["yrdln"].median())

# 3. Time Securities: Fill remaining time gaps using forward-fill
valid_plays["TimeSecs"] = valid_plays.groupby("GameID")["TimeSecs"].ffill().bfill()
valid_plays["PlayTimeDiff"] = valid_plays["PlayTimeDiff"].fillna(
    valid_plays["PlayTimeDiff"].median()
)

# 4. Categorical / Team Identifiers: Fill unassigned posteam/DefensiveTeam with 'UNKNOWN'
valid_plays["posteam"] = valid_plays["posteam"].fillna("UNKNOWN")
valid_plays["DefensiveTeam"] = valid_plays["DefensiveTeam"].fillna("UNKNOWN")
valid_plays["SideofField"] = valid_plays["SideofField"].fillna("MID")

# Verify remaining null count across core features
print(
    "Remaining Nulls in core features:\n",
    valid_plays[
        [
            "down",
            "yrdline100",
            "TimeSecs",
            "posteam",
            "DefensiveTeam",
            "Yards.Gained",
        ]
    ].isnull().sum(),
)

# Deduplication & Text Normalization
# Remove duplicate records if any exist
initial_rows = len(valid_plays)
valid_plays.drop_duplicates(inplace=True)
print(f"Removed {initial_rows - len(valid_plays)} duplicate records.")

# Standardize String Fields (Trim whitespaces, convert to uppercase)
string_cols = ["posteam", "DefensiveTeam", "SideofField", "PlayType"]
for col in string_cols:
    if col in valid_plays.columns:
        valid_plays[col] = valid_plays[col].astype(str).str.strip().str.upper()

# Normalize PlayType entries (Example mapping inconsistent labels)
playtype_clean_map = {
    "PASS": "PASS",
    "RUN": "RUSH",
    "RUSH": "RUSH",
    "PUNT": "PUNT",
    "KICKOFF": "KICKOFF",
    "SACK": "SACK",
    "FIELD GOAL": "FIELD_GOAL",
    "EXTRA POINT": "EXTRA_POINT",
    "NO PLAY": "NO_PLAY",
}
valid_plays["PlayType"] = (
    valid_plays["PlayType"].map(playtype_clean_map).fillna("OTHER")
)

print("Standardized PlayType Categories:\n", valid_plays["PlayType"].value_counts())

# Data Type Casting
# Convert Dates to proper datetime64[ns]
valid_plays["Date"] = pd.to_datetime(valid_plays["Date"], format="%Y-%m-%d")

# Convert Game IDs and categorical tags to string/object
valid_plays["GameID"] = valid_plays["GameID"].astype(str)
valid_plays["Drive"] = valid_plays["Drive"].astype(int)
valid_plays["qtr"] = valid_plays["qtr"].astype(int)
valid_plays["down"] = valid_plays["down"].astype(int)

# Numeric metrics to standardized float/int
numeric_conversions = [
    "Yards.Gained",
    "yrdline100",
    "ydstogo",
    "ydsnet",
    "TimeSecs",
    "PlayTimeDiff",
]
for col in numeric_conversions:
    if col in valid_plays.columns:
        valid_plays[col] = pd.to_numeric(valid_plays[col], errors="coerce")

print("Verified Data Types:")
print(
    valid_plays[
        ["Date", "GameID", "down", "yrdline100", "PlayType", "Yards.Gained"]
    ].dtypes
)

# IQR Outlier Detection for Continuous Numeric Fields
numeric_cols = ["Yards.Gained", "PlayTimeDiff", "ydstogo"]

outlier_summary = []

for col in numeric_cols:
    Q1 = valid_plays[col].quantile(0.25)
    Q3 = valid_plays[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (valid_plays[col] < lower_bound) | (valid_plays[col] > upper_bound)
    ).sum()

    outlier_summary.append(
        {
            "Feature": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "Lower_Limit": round(lower_bound, 2),
            "Upper_Limit": round(upper_bound, 2),
            "Outlier_Count": outlier_count,
        }
    )

print(pd.DataFrame(outlier_summary))

# Decision & Capping on Yards.Gained:
# Yards gained cannot exceed total remaining field yards (100) or be less than -50 (extreme sacks/fumbles)
valid_plays["Yards.Gained"] = valid_plays["Yards.Gained"].clip(
    lower=-40.0, upper=99.0
)

# Before vs. After Summary Comparison
summary_table = pd.DataFrame(
    {
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Total Missing Values",
            "Duplicate Rows",
            "Date Dtype",
            "Sparse Columns (>90% Nulls)",
        ],
        "Before Cleaning (Raw)": [
            raw_df.shape[0],
            raw_df.shape[1],
            int(raw_df.isnull().sum().sum()),
            int(raw_df.duplicated().sum()),
            str(raw_df["Date"].dtype),
            len(sparse_cols),
        ],
        "After Cleaning (Transformed)": [
            valid_plays.shape[0],
            valid_plays.shape[1],
            int(valid_plays.isnull().sum().sum()),
            int(valid_plays.duplicated().sum()),
            str(valid_plays["Date"].dtype),
            0,
        ],
    }
)

print(summary_table)

# Save the final cleaned dataframe to CSV
cleaned_filename = "cleaned_nfl_play_by_play.csv"
valid_plays.to_csv(cleaned_filename, index=False)
print(f"\nCleaned dataset successfully saved to: {cleaned_filename}")