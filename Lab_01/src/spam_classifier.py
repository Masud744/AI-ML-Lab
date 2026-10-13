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
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def plot_visuals(df, y_test, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Class Distribution
    plt.figure(figsize=(5, 3.8))
    counts = df["label"].value_counts()
    bars = plt.bar(counts.index, counts.values, color=["#3470a3", "#c94a4a"], width=0.45, edgecolor="black")
    plt.title("Class Distribution in Dataset", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Class", fontsize=10)
    plt.ylabel("Number of Messages", fontsize=10)
    plt.ylim(0, 5700)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2, h + 50, f"{h}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    fig, ax = plt.subplots(figsize=(4.8, 4))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / np.sum(cm[i, :])) * 100
            txt_color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")

    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham (Pred)", "Spam (Pred)"], fontsize=9)
    ax.set_yticklabels(["Ham (Actual)", "Spam (Actual)"], fontsize=9)
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    data_path = os.path.join(project_root, "dataset", "SMSSpamCollection.txt")
    output_dir = os.path.join(project_root, "submission_report", "screenshots")

    # Load dataset
    df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "message"])
    print(f"Loaded dataset: {len(df)} messages ({df['label'].value_counts()['ham']} ham, {df['label'].value_counts()['spam']} spam)")

    # Preprocess text
    df["clean_message"] = df["message"].apply(preprocess_text)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    # Text vectorization using Bag of Words
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} words")

    # Train Naive Bayes model
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam")
    rec = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")

    print(f"\n--- Model Performance ---\nAccuracy  : {acc * 100:.2f}%\nPrecision : {prec * 100:.2f}% (Spam)\nRecall    : {rec * 100:.2f}% (Spam)\nF1-Score  : {f1 * 100:.2f}% (Spam)")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'])}")

    # Test sample message
    test_message = "Congratulations! You have won a free ticket. Call now!"
    cleaned_test = preprocess_text(test_message)
    test_vec = vectorizer.transform([cleaned_test])
    prediction = model.predict(test_vec)[0]
    probabilities = model.predict_proba(test_vec)[0]
    spam_idx = list(model.classes_).index("spam")
    ham_idx = list(model.classes_).index("ham")

    print(f"--- Test Query Prediction ---\nInput text : {test_message}\nPrediction : {prediction.upper()}\nProbability: Ham = {probabilities[ham_idx] * 100:.2f}%, Spam = {probabilities[spam_idx] * 100:.2f}%")

    # Save visual plots for report
    plot_visuals(df, y_test, y_pred, output_dir)
    print("\nPlots saved to screenshots directory successfully.")


if __name__ == "__main__":
    main()
