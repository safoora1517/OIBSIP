# Environment Setup & Library Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Styling configurations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

# Load Dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\OIBSIP\\DataAnalytics-L2-HousePricePrediction\\House Price Prediction Dataset.csv")
print(f"Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()

# Structural Inspection & Missing Value Audit
print("--- Data Types & Non-Null Values ---")
print(df.info())

print("\n--- Missing Values Count per Column ---")
null_counts = df.isnull().sum()
print(null_counts)

print("\n--- Descriptive Statistics for Numerical Variables ---")
print(df.describe().T.round(2))

# Target Variable (Price) Distribution Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram with KDE
sns.histplot(df["Price"], kde=True, color="#2b5c8f", bins=30, ax=axes[0])
axes[0].set_title("Distribution of House Prices", fontweight="bold")
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Frequency")

# Boxplot to inspect potential outliers
sns.boxplot(x=df["Price"], color="#7293cb", ax=axes[1])
axes[1].set_title("Boxplot of House Prices", fontweight="bold")
axes[1].set_xlabel("Price ($)")

plt.tight_layout()
plt.show()

# One-Hot Encoding & Preprocessing
# Drop non-predictive identifier column
clean_df = df.drop(columns=["Id"]).copy()

# One-Hot Encode categorical features with drop_first=True to avoid dummy variable trap
encoded_df = pd.get_dummies(
    clean_df,
    columns=["Location", "Condition", "Garage"],
    drop_first=True,
    dtype=int,
)

print(
    f"Preprocessed Dataset Shape: {encoded_df.shape[0]} rows, {encoded_df.shape[1]} features"
)
encoded_df.head()

# Correlation Matrix
plt.figure(figsize=(12, 8))
corr_matrix = encoded_df.corr()

# Mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-0.15,
    vmax=0.15,
    linewidths=0.5,
)
plt.title(
    "Correlation Heatmap (Predictor Features vs. Price)", fontweight="bold"
)
plt.tight_layout()
plt.show()

# Sorted linear correlation with target variable Price
print("--- Linear Correlation with Price ---")
print(corr_matrix["Price"].sort_values(ascending=False))

# Train/Test Split (80/20)
X = encoded_df.drop(columns=["Price"])
y = encoded_df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training feature shape: {X_train.shape}")
print(f"Testing feature shape:  {X_test.shape}")

# Fit Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Generate Predictions
y_pred_lr = lr_model.predict(X_test)

# Model Evaluation Metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

eval_summary = pd.DataFrame(
    {
        "Evaluation Metric": [
            "Mean Absolute Error (MAE)",
            "Mean Squared Error (MSE)",
            "Root Mean Squared Error (RMSE)",
            "R-squared Score (R²)",
        ],
        "Score": [
            f"${mae_lr:,.2f}",
            f"{mse_lr:,.2f}",
            f"${rmse_lr:,.2f}",
            f"{r2_lr:.4f}",
        ],
    }
)

print(eval_summary)# Cell 7: Model Evaluation Metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

eval_summary = pd.DataFrame(
    {
        "Evaluation Metric": [
            "Mean Absolute Error (MAE)",
            "Mean Squared Error (MSE)",
            "Root Mean Squared Error (RMSE)",
            "R-squared Score (R²)",
        ],
        "Score": [
            f"${mae_lr:,.2f}",
            f"{mse_lr:,.2f}",
            f"${rmse_lr:,.2f}",
            f"{r2_lr:.4f}",
        ],
    }
)

print(eval_summary)

# Actual vs. Predicted Prices Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=y_test, y=y_pred_lr, alpha=0.6, color="#1f77b4", edgecolor="none", s=50
)

# Reference 45-degree diagonal line
min_val = min(y_test.min(), y_pred_lr.min())
max_val = max(y_test.max(), y_pred_lr.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Ideal Prediction Line",
)

plt.title("Actual vs. Predicted House Prices", fontweight="bold")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.legend()
plt.tight_layout()
plt.show()

# Residual Diagnostic Plot
residuals = y_test - y_pred_lr

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Residuals vs Predicted
sns.scatterplot(
    x=y_pred_lr,
    y=residuals,
    alpha=0.6,
    color="#e6550d",
    edgecolor="none",
    s=50,
    ax=axes[0],
)
axes[0].axhline(0, color="black", linestyle="--", linewidth=1.5)
axes[0].set_title("Residuals vs. Fitted Values", fontweight="bold")
axes[0].set_xlabel("Fitted Values (Predicted $)")
axes[0].set_ylabel("Residuals ($)")

# Residual Distribution Check
sns.histplot(residuals, kde=True, color="#31a354", bins=25, ax=axes[1])
axes[1].set_title("Distribution of Residual Errors", fontweight="bold")
axes[1].set_xlabel("Residual Error ($)")

plt.tight_layout()
plt.show()

# Feature Impact Analysis
coef_df = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": lr_model.coef_}
).sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=coef_df,
    x="Coefficient",
    y="Feature",
    palette="vlag",
    orient="h",
)
plt.title(
    "Feature Coefficients: Direction & Magnitude of Impact on Price",
    fontweight="bold",
)
plt.xlabel("Coefficient Value ($ per unit change)")
plt.ylabel("Feature")
plt.axvline(0, color="grey", linestyle="--")
plt.tight_layout()
plt.show()

print(coef_df)

# Cell 9: Model Benchmark with L1 & L2 Regularization
# Train Ridge (L2)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Train Lasso (L1)
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Comparison Table
comparison_df = pd.DataFrame(
    {
        "Model": ["Ordinary Least Squares (OLS)", "Ridge Regression (L2)", "Lasso Regression (L1)"],
        "RMSE ($)": [
            f"${np.sqrt(mean_squared_error(y_test, y_pred_lr)):,.2f}",
            f"${np.sqrt(mean_squared_error(y_test, y_pred_ridge)):,.2f}",
            f"${np.sqrt(mean_squared_error(y_test, y_pred_lasso)):,.2f}",
        ],
        "R² Score": [
            f"{r2_score(y_test, y_pred_lr):.4f}",
            f"{r2_score(y_test, y_pred_ridge):.4f}",
            f"{r2_score(y_test, y_pred_lasso):.4f}",
        ],
    }
)

print(comparison_df)