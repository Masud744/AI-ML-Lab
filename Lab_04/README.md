# Lab 04: Breast Cancer Classification using Ensemble Learning Techniques

Course Code: AI 484  
Course Title: Artificial Intelligence and Machine Learning Sessional  
Department: Department of Internet of Things and Robotics Engineering (IRE)  
Faculty: Faculty of Cyber Physical System Engineering  
Institution: University of Frontier Technology, Bangladesh (UFTB)  

Author: Shahriar Alom Masud  
Student ID: 2201032  
Session: 2022-2023  
Instructor: Suman Saha, Assistant Professor, Dept. of IRE, UFTB  

---

## Problem Statement & Objectives

Given the Breast Cancer Wisconsin Diagnostic dataset, evaluate and compare four primary ensemble learning paradigms to accurately classify tumors as **Benign** or **Malignant**:
1. **Bagging Classifier**: Bootstrap aggregating using Decision Trees as base estimators.
2. **Boosting Classifier (AdaBoost)**: Sequential training of weak decision stumps with instance weight updating.
3. **Stacking Classifier**: Combining heterogeneous base learners (Decision Tree, KNN, Support Vector Machine) with a Logistic Regression meta-classifier.
4. **Random Forest Classifier**: Multiple decision trees with bootstrap sampling and random feature subspace selection.

### Key Tasks:
- Load Wisconsin Diagnostic dataset and conduct exploratory analysis.
- Verify missing values (confirmed 0 missing values across all 569 instances and 30 cytological features).
- Preprocess data: binary label encoding (Malignant $\rightarrow$ 1, Benign $\rightarrow$ 0) and standard Z-score feature scaling (`StandardScaler`).
- Stratified 80-20 train-test split (455 training, 114 holdout test samples).
- Train, evaluate, and inspect model structure (estimator contributions, decision tree visualization, feature importance, Stacking meta-learner weights).
- Assess performance across standard metrics (Accuracy, Precision, Recall, F1-Score) and identify the clinically optimal model.

---

## Experimental Results

Evaluation on the holdout test set (114 samples: 72 Benign, 42 Malignant):

| Ensemble Model | Accuracy | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bagging** (50 Estimators) | 97.37% | 100.00% | 92.86% | 96.30% | 72 | 0 | 3 | 39 |
| **Boosting (AdaBoost)** (50 Stumps) | **98.25%** | **100.00%** | **95.24%** | **97.56%** | 72 | 0 | 2 | 40 |
| **Stacking** (DT + KNN + SVM) | 96.49% | 97.50% | 92.86% | 95.12% | 71 | 1 | 3 | 39 |
| **Random Forest** (100 Trees) | 97.37% | 100.00% | 92.86% | 96.30% | 72 | 0 | 3 | 39 |

### Architectural Insights & Clinical Justification:
- **Optimal Model**: **AdaBoost** achieved the best overall performance with **98.25% accuracy** and the highest recall (**95.24%**), missing only 2 malignant cases out of 42. In oncology, minimizing False Negatives (FN) is paramount because an undetected malignancy can be fatal.
- **Precision**: Bagging, AdaBoost, and Random Forest all achieved **100.00% precision** (zero false alarms among the 72 benign samples).
- **Stacking Meta-Learner Weights**:
  - SVM Coefficient: `2.7852`
  - KNN Coefficient: `1.7273`
  - Decision Tree Coefficient: `0.8700`
  - Demonstrates that maximum-margin hyperplane separation provided the strongest discriminative meta-features.
- **Top Nuclear Morphology Predictors**: `area_worst` (0.151), `concave_points_worst` (0.126), `radius_worst` (0.093), and `perimeter_worst` (0.084) account for over 45% of cumulative decision splits.

---

## Project Structure

```text
Lab_04/
├── dataset/
│   └── breast_cancer.csv
├── src/
│   └── cancer_ensemble_classifier.py
├── submission_report/
│   ├── build.sh
│   ├── main.tex
│   ├── 2201032_Report_4.pdf
│   └── screenshots/
│       ├── class_distribution.png
│       ├── confusion_matrices.png
│       ├── feature_importance.png
│       ├── metrics_comparison.png
│       ├── tree_visualization.png
│       └── uftb_logo.png
├── Lab 4.pdf
└── README.md
```

---

## Running the Code

### 1. Run Ensemble Classifier Script
```bash
python3 src/cancer_ensemble_classifier.py
```

### 2. Build PDF Lab Report
```bash
cd submission_report
bash build.sh
```
The compiled PDF report is located at:
`submission_report/2201032_Report_4.pdf`

