# Environment Setup & Library Imports
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

# Styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Download NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Load Dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\OIBSIP\\DataAnalytics-L1-SentimentAnalysis\\test.csv", encoding="latin-1")

# Data Quality Inspection & Class Distribution
print("Raw Dataset Dimensions:", df.shape)
print("\n--- Missing Values Check ---")
print(df[["text", "sentiment"]].isnull().sum())

# Filter out null rows in target and text
df = df.dropna(subset=["text", "sentiment"]).copy()
df["sentiment"] = df["sentiment"].str.strip().str.lower()

print("\nCleaned Row Count:", len(df))
print("\nSentiment Class Distribution:")
print(df["sentiment"].value_counts())

# Sentiment Distribution Bar Chart
plt.figure(figsize=(8, 5))
sentiment_counts = df["sentiment"].value_counts()
colors = ["#4C72B0", "#55A868", "#C44E52"]

ax = sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    palette=colors,
    hue=sentiment_counts.index,
    legend=False,
)
plt.title("Sentiment Class Distribution in Dataset", fontweight="bold")
plt.xlabel("Sentiment Class")
plt.ylabel("Number of Tweets / Reviews")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
    )

plt.tight_layout()
plt.show()

# Text Preprocessing Pipeline
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # 1. Lowercase
    text = str(text).lower()
    # 2. Remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    # 3. Remove punctuation and non-alphabetical characters
    text = re.sub(r"[^a-z\s]", "", text)
    # 4. Tokenization & Lemmatization without stopwords
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)


df["cleaned_text"] = df["text"].apply(preprocess_text)
df[["text", "cleaned_text", "sentiment"]].head()

# Word Cloud Visualisations per Sentiment Class
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
classes = ["positive", "neutral", "negative"]
cmaps = ["Greens", "Blues", "Reds"]

for ax, sent, cmap in zip(axes, classes, cmaps):
    text_corpus = " ".join(df[df["sentiment"] == sent]["cleaned_text"])
    wc = WordCloud(
        width=500,
        height=400,
        background_color="white",
        colormap=cmap,
        max_words=100,
    ).generate(text_corpus)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Top Words: {sent.capitalize()} Sentiment", fontweight="bold")

plt.tight_layout()
plt.show()

# Train / Test Split & Feature Vectorization
X = df["cleaned_text"]
y = df["sentiment"]

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Extraction: TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print(
    f"Training Feature Matrix: {X_train_vec.shape[0]} rows, {X_train_vec.shape[1]} features"
)
print(f"Testing Feature Matrix:  {X_test_vec.shape[0]} rows")

# Model Training (Naive Bayes vs. Logistic Regression)
# Model 1: Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)

# Model 2: Logistic Regression (Hyperparameter Tuned)
lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)

# Model Evaluation & Metric Comparison
models = {
    "Multinomial Naive Bayes": (y_pred_nb, nb_model),
    "Logistic Regression": (y_pred_lr, lr_model),
}

eval_list = []
for name, (pred, model) in models.items():
    eval_list.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (Macro)": precision_score(
                y_test, pred, average="macro", zero_division=0
            ),
            "Recall (Macro)": recall_score(
                y_test, pred, average="macro", zero_division=0
            ),
            "F1-Score (Macro)": f1_score(
                y_test, pred, average="macro", zero_division=0
            ),
        }
    )

eval_df = pd.DataFrame(eval_list).round(4)
display(eval_df)

print("\n--- Detailed Classification Report: Logistic Regression ---")
print(classification_report(y_test, y_pred_lr))

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = ["negative", "neutral", "positive"]

# Naive Bayes Heatmap
cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels)
sns.heatmap(
    cm_nb,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix: Naive Bayes", fontweight="bold")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Logistic Regression Heatmap
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[1],
)
axes[1].set_title("Confusion Matrix: Logistic Regression", fontweight="bold")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.show()

# Error Analysis (Inspecting 5 Misclassifications)
test_results = pd.DataFrame(
    {
        "Original_Text": df.loc[X_test.index, "text"],
        "Cleaned_Text": X_test,
        "Actual_Sentiment": y_test,
        "Predicted_Sentiment": y_pred_lr,
    }
)

misclassified = test_results[
    test_results["Actual_Sentiment"] != test_results["Predicted_Sentiment"]
]

print(f"Total Misclassified Samples: {len(misclassified)}")
print("\n--- 5 Characteristic Misclassified Samples ---")
for idx, row in misclassified.head(5).iterrows():
    print(f"• Text: \"{row['Original_Text']}\"")
    print(f"  Actual: [{row['Actual_Sentiment'].upper()}] | Predicted: [{row['Predicted_Sentiment'].upper()}]\n")