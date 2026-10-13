# Lab 01: SMS Spam Detection using Naive Bayes Classifier

**Course Code:** AI 484  
**Course Title:** Artificial Intelligence and Machine Learning Sessional  
**Department:** Department of Internet of Things and Robotics Engineering (IRE)  
**Faculty:** Faculty of Cyber Physical System Engineering  
**Institution:** University of Frontier Technology, Bangladesh (UFTB)  

**Author:** Shahriar Alom Masud  
**Student ID:** 2201032  
**Session:** 2022–2023  
**Instructor:** Suman Saha, Assistant Professor, Dept. of IRE, UFTB  

---

## 📌 Problem Formulation & Tasks

Given the **SMS Spam Collection dataset**, how can you use the **Naive Bayes classifier** to automatically detect whether an incoming SMS is spam or not?

### Key Tasks:
1. **Load the dataset:** Ingest the SMS Spam Collection corpus containing 5,572 tagged messages.
2. **Preprocess text:** Normalize characters to lowercase, eliminate punctuation and numbers, and vectorize into Bag-of-Words features using `CountVectorizer`.
3. **Train Naive Bayes classifier:** Train a `MultinomialNB` model with Laplace smoothing ($\alpha = 1.0$) on an 80-20 stratified train-test split.
4. **Evaluate performance:** Compute Accuracy, Precision, Recall, F1-Score, and generate the Confusion Matrix.
5. **Inference on assigned test prompt:** Classify the incoming message:
   > *"Congratulations! You have won a free ticket. Call now!"*

---

## 📊 Experimental Results

| Metric | Spam Class | Overall Benchmark |
| :--- | :---: | :---: |
| **Accuracy** | — | **98.21%** |
| **Precision** | **97.78%** | **98.00%** |
| **Recall** | **88.59%** | **94.00%** |
| **F1-Score** | **92.96%** | **96.00%** |

### Assigned Query Prediction:
* **Input Text:** `"Congratulations! You have won a free ticket. Call now!"`
* **Cleaned Text:** `"congratulations you have won a free ticket call now"`
* **Predicted Class:** **`SPAM`**
* **Posterior Probabilities:**
  * **Spam:** `99.9868%`
  * **Ham:** `0.0132%`

---

## 📁 Directory Structure

```text
Lab_01/
├── dataset/
│   └── SMSSpamCollection.txt          # SMS Spam Collection dataset (5,572 records)
├── src/
│   └── spam_classifier.py             # Complete Python pipeline & visualizer
├── submission_report/
│   ├── build.sh                       # 1-Click LaTeX compilation script
│   ├── main.tex                       # LaTeX report source code
│   ├── 2201032_Report_1.pdf           # Compiled lab report PDF
│   └── screenshots/                   # Exported graphs and UFTB logo
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       ├── metrics_chart.png
│       └── uftb_logo.png
└── README.md
```

---

## 🚀 How to Run

### 1. Run Python Classifier & Export Figures:
```bash
python3 src/spam_classifier.py
```

### 2. Compile LaTeX Lab Report to PDF:
```bash
cd submission_report
bash build.sh
```
The compiled PDF will be generated at:
`submission_report/2201032_Report_1.pdf`
