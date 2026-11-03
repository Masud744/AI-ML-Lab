import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)

    # Wisconsin dataset has 'id' and 'diagnosis' followed by 30 cytological measurements
    feature_cols = [c for c in df.columns if c not in ["id", "diagnosis"]]
    X = df[feature_cols].copy()
    y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)

    # Missing value verification
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Distance-based base learners in Stacking (KNN, SVM) require zero-mean unit-variance scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    return df, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


def build_models():
    base_tree = DecisionTreeClassifier(random_state=42)

    bagging = BaggingClassifier(
        estimator=base_tree,
        n_estimators=50,
        bootstrap=True,
        random_state=42
    )

    boosting = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )

    stacking = StackingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("svm", SVC(kernel="rbf", random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        bootstrap=True,
        random_state=42
    )

    return {
        "Bagging": bagging,
        "Boosting (AdaBoost)": boosting,
        "Stacking": stacking,
        "Random Forest": rf
    }


def plot_visuals(df, models, X_test, y_test, feature_cols, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Class Distribution
    plt.figure(figsize=(4.8, 3.8))
    counts = df["diagnosis"].value_counts()
    labels = ["Benign (B)", "Malignant (M)"]
    vals = [counts["B"], counts["M"]]
    bars = plt.bar(labels, vals, color=["#2b6cb0", "#c53030"], width=0.45, edgecolor="#2d3748")
    plt.title("Tumor Class Distribution", fontsize=11, fontweight="bold", pad=12)
    plt.ylabel("Number of Samples", fontsize=10)
    plt.ylim(0, 420)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{h}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.8))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ax = axes[idx]
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = (val / np.sum(cm[i, :])) * 100
                txt_color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, fontsize=9.5, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Malignant"], fontsize=8.5)
        ax.set_yticklabels(["Benign", "Malignant"], fontsize=8.5)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        ax.set_title(name, fontsize=10.5, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Feature Importance (Random Forest)
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(6.2, 4.0))
    top_features = [feature_cols[i] for i in indices][::-1]
    top_importances = importances[indices][::-1]

    plt.barh(top_features, top_importances, color="#319795", edgecolor="#234e52", height=0.6)
    plt.title("Top 10 Feature Importances (Random Forest)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Gini Importance Score", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    for i, v in enumerate(top_importances):
        plt.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Individual Decision Tree Visualization from Ensemble
    sample_tree = rf_model.estimators_[0]
    plt.figure(figsize=(14, 6.5))
    plot_tree(
        sample_tree,
        max_depth=3,
        feature_names=feature_cols,
        class_names=["Benign", "Malignant"],
        filled=True,
        rounded=True,
        fontsize=7
    )
    plt.title("Representative Decision Tree Structure (Pruned to Depth 3)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tree_visualization.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(metrics_dict, output_dir):
    names = list(metrics_dict.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(names))
    width = 0.18

    plt.figure(figsize=(7.5, 4.2))
    colors = ["#2b6cb0", "#2f855a", "#dd6b20", "#6b46c1"]

    for i, m_name in enumerate(metric_names):
        vals = [metrics_dict[name][m_name] * 100 for name in names]
        rects = plt.bar(x + (i - 1.5) * width, vals, width, label=m_name, color=colors[i], edgecolor="#1a202c")
        for rect in rects:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=6.8)

    plt.title("Ensemble Models Performance Comparison", fontsize=11, fontweight="bold", pad=12)
    plt.xticks(x, names, fontsize=9.5)
    plt.ylabel("Score (%)", fontsize=10)
    plt.ylim(80, 108)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    data_path = os.path.join(project_root, "dataset", "breast_cancer.csv")
    output_dir = os.path.join(project_root, "submission_report", "screenshots")

    df, X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess_data(data_path)

    models = build_models()
    metrics_summary = {}

    print(f"Dataset: {len(df)} records ({df['diagnosis'].value_counts()['B']} Benign, {df['diagnosis'].value_counts()['M']} Malignant)")
    print(f"Holdout Test: {len(y_test)} records (Benign: {sum(y_test==0)}, Malignant: {sum(y_test==1)})\n")
    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'TN':<3} {'FP':<3} {'FN':<3} {'TP':<3}")
    print("-" * 84)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        metrics_summary[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Confusion Matrix": cm
        }

        print(f"{name:<22} | {acc*100:6.2f}% | {prec*100:7.2f}% | {rec*100:6.2f}% | {f1*100:6.2f}% | {cm[0,0]:<3} {cm[0,1]:<3} {cm[1,0]:<3} {cm[1,1]:<3}")

    plot_visuals(df, models, X_test, y_test, feature_cols, output_dir)
    plot_metrics_comparison(metrics_summary, output_dir)

    stack_lr = models["Stacking"].final_estimator_
    print("\nStacking Meta-Learner Logistic Regression Coefficients:")
    for estimator_name, coef in zip(["Decision Tree", "KNN", "SVM"], stack_lr.coef_[0]):
        print(f"  - {estimator_name:<15}: {coef:.4f}")
    print(f"  - Intercept      : {stack_lr.intercept_[0]:.4f}")


if __name__ == "__main__":
    main()

