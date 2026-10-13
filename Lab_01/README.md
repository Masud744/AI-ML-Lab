# Lab 01: SMS Spam Detection using Naive Bayes Classifier

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

Given the SMS Spam Collection dataset, use the Naive Bayes classifier to automatically detect whether an incoming SMS is spam or not.

### Objectives:
1. Load and explore the SMS Spam Collection dataset (5,572 messages).
2. Clean and preprocess the raw text (lowercasing, punctuation removal, tokenization).
3. Extract numerical feature vectors using Bag-of-Words (CountVectorizer).
4. Train a Multinomial Naive Bayes model using an 80-20 stratified train-test split.
5. Evaluate model performance using Accuracy, Precision, Recall, and Confusion Matrix.
6. Test prediction on the assigned unseen sample:
   "Congratulations! You have won a free ticket. Call now!"

---

## Experimental Results

Evaluation on the test set (1,115 messages: 966 ham, 149 spam):

| Metric | Score | Note |
| :--- | :---: | :--- |
| Accuracy | 98.48% | Overall test accuracy |
| Precision (Spam) | 97.83% | Only 3 false alarms among 966 legitimate messages |
| Recall (Spam) | 90.60% | Successfully detected 135 out of 149 spam messages |
| F1-Score (Spam) | 94.08% | Harmonic mean of precision and recall |

### Test Query Output:
- Input Message: "Congratulations! You have won a free ticket. Call now!"
- Predicted Class: SPAM
- Class Probability:
  - Spam: 99.89%
  - Ham: 0.11%

---

## Project Structure

```text
Lab_01/
├── dataset/
│   └── SMSSpamCollection.txt
├── src/
│   └── spam_classifier.py
├── submission_report/
│   ├── build.sh
│   ├── main.tex
│   ├── 2201032_Report_1.pdf
│   └── screenshots/
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       └── uftb_logo.png
└── README.md
```

---

## Running the Code

### 1. Run Classifier Script
```bash
python3 src/spam_classifier.py
```

### 2. Build PDF Lab Report
```bash
cd submission_report
bash build.sh
```
The compiled PDF report will be generated at:
`submission_report/2201032_Report_1.pdf`
