#!/usr/bin/env python3
"""
Lab 01: SMS Spam Detection using Naive Bayes Classifier
Course: AI 484 - Artificial Intelligence and Machine Learning Sessional
Department of Internet of Things and Robotics Engineering (IRE)
University of Frontier Technology, Bangladesh (UFTB)

Author: Shahriar Alom Masud
Student ID: 2201032
Session: 2022-2023
"""

import os
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ==============================================================================
# 1. TEXT PREPROCESSING FUNCTION
# ==============================================================================
def clean_text(text: str) -> str:
    """Preprocesses input text:

    - Converts to lowercase
    - Removes punctuation and digits
    - Removes excessive whitespaces
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers / digits
    text = re.sub(r"\d+", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# 2. DATASET LOADER
# ==============================================================================
def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads the SMS Spam Collection dataset from tab-separated file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "message"])
    return df


# ==============================================================================
# 3. VISUALIZATION EXPORTERS
# ==============================================================================
def export_visualizations(df, y_test, y_pred, output_dir: str):
    """Exports high-resolution figures for LaTeX lab report."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    counts = df["label"].value_counts()
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax.set_title("SMS Spam Collection: Class Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Message Count", fontsize=10, fontweight="bold")
    ax.set_xlabel("Class Label", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bar counts and percentages
    total = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "class_distribution.png")
    fig.savefig(dist_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved class distribution plot: {dist_path}")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=300)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / np.sum(cm[i, :])) * 100
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10.5, fontweight="bold")

    fig.colorbar(cax, fraction=0.046, pad=0.05)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham (Pred)", "Spam (Pred)"], fontsize=9.5, fontweight="bold")
    ax.set_yticklabels(["Ham (Actual)", "Spam (Actual)"], fontsize=9.5, fontweight="bold")
    ax.set_title("Confusion Matrix (MultinomialNB)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved confusion matrix: {cm_path}")

    # 3. Performance Metrics Bar Chart
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam")
    rec = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_values = [acc * 100, prec * 100, rec * 100, f1 * 100]
    metric_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    bars = ax.bar(metric_names, metric_values, color=metric_colors, width=0.55, edgecolor="black", linewidth=1.1)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_title("Model Performance Metrics (Spam Class)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, "metrics_chart.png")
    fig.savefig(metrics_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved metrics chart: {metrics_path}")


# ==============================================================================
# 4. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    dataset_path = os.path.join(project_root, "dataset", "SMSSpamCollection.txt")
    screenshots_dir = os.path.join(project_root, "submission_report", "screenshots")

    print("==================================================================")
    print(" AI 484: SMS Spam Detection using Naive Bayes Classifier")
    print("==================================================================")

    # 1. Load Dataset
    print("\n[Step 1] Loading SMS Spam Collection dataset...")
    df = load_dataset(dataset_path)
    print(f"Total rows: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    # 2. Preprocess Text
    print("\n[Step 2] Preprocessing message texts...")
    df["cleaned_message"] = df["message"].apply(clean_text)

    # 3. Train-Test Split (80% Train, 20% Test with Stratification)
    print("\n[Step 3] Splitting dataset into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4. Vectorization (CountVectorizer / Bag-of-Words)
    print("\n[Step 4] Vectorizing text with CountVectorizer (Bag of Words)...")
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} distinct tokens")

    # 5. Train Naive Bayes Classifier
    print("\n[Step 5] Training Multinomial Naive Bayes classifier...")
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X_train_vec, y_train)
    print("Model training completed successfully.")

    # 6. Evaluation on Unseen Test Set
    print("\n[Step 6] Evaluating model on unseen test set...")
    y_pred = classifier.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam")
    rec = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")

    print("\n" + "=" * 50)
    print(" EVALUATION METRICS:")
    print("=" * 50)
    print(f" Accuracy : {acc * 100:.2f}%")
    print(f" Precision: {prec * 100:.2f}% (Spam)")
    print(f" Recall   : {rec * 100:.2f}% (Spam)")
    print(f" F1-Score : {f1 * 100:.2f}% (Spam)")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # 7. Predict on Required Test Query
    print("=" * 50)
    print(" INFERENCE ON REQUIRED LAB TEST QUERY:")
    print("=" * 50)
    test_query = "Congratulations! You have won a free ticket. Call now!"
    cleaned_query = clean_text(test_query)
    query_vec = vectorizer.transform([cleaned_query])
    prediction = classifier.predict(query_vec)[0]
    probabilities = classifier.predict_proba(query_vec)[0]
    class_indices = {cls: idx for idx, cls in enumerate(classifier.classes_)}

    print(f"Input Message   : \"{test_query}\"")
    print(f"Cleaned Message : \"{cleaned_query}\"")
    print(f"Predicted Class : {prediction.upper()}")
    print(f"Probability [Ham]  : {probabilities[class_indices['ham']] * 100:.4f}%")
    print(f"Probability [Spam] : {probabilities[class_indices['spam']] * 100:.4f}%")

    # 8. Export Visualizations for LaTeX Report
    print("\n[Step 7] Exporting visualizations for LaTeX report...")
    export_visualizations(df, y_test, y_pred, screenshots_dir)
    print("\nExecution finished successfully!")


if __name__ == "__main__":
    main()
