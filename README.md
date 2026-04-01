# 🧠 ML Model Optimization & Evaluation — Customer Churn & Credit Risk

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ECC71?style=for-the-badge)
![Course](https://img.shields.io/badge/Course-DTSC%205082-blueviolet?style=for-the-badge)

**End-to-end machine learning pipeline spanning 3 real-world datasets — with hyperparameter tuning, cross-validation, and systematic model evaluation.**

[📌 Overview](#-overview) • [📦 Datasets](#-datasets) • [🔬 Methods](#-methods) • [📈 Results](#-results) • [🚀 Getting Started](#-getting-started)

</div>

---

## 📌 Overview

Building a model is only the first step. The real challenge in production ML is **optimization** — tuning hyperparameters, avoiding overfitting, and systematically evaluating performance across meaningful metrics.

This project (Phase 3 of an ongoing research series) focuses on the optimization and rigorous evaluation of classification models applied to **customer churn prediction** and **credit risk assessment** across three industry-standard datasets.

---

## 📦 Datasets

| Dataset | Domain | Key Task |
|---|---|---|
| **Telco Customer Churn** | Telecommunications | Predict customer churn (Yes/No) |
| **Credit Card Default** | Finance | Predict credit default (UCI ML Repo) |
| **German Credit** | Finance | Classify credit risk (Good/Bad) |

### Dataset Details

**Telco Customer Churn**
- 7,032 rows after cleaning
- Features: tenure, monthly charges, contract type, internet service, payment method
- Target: `Churn` (binary)

**Credit Card Default** (UCI)
- 30,000 records
- Features: payment history, bill amounts, demographic info
- Target: `default payment next month`

**German Credit** (UCI Statlog)
- 1,000 records, 20 features
- Classic benchmark for binary credit classification

---

## 🔬 Methods

### Preprocessing Pipeline
```python
# Key steps applied
1. Drop irrelevant ID columns (customerID)
2. Convert target to numeric (Yes → 1, No → 0)
3. Coerce TotalCharges to float, drop nulls
4. One-hot encode all categorical features
5. Stratified train/test split (80/20)
```

### Models Evaluated
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

### Optimization Strategy
```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning with cross-validation
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
```

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve
- Cross-validation scores (5-fold)

---

## 📈 Results

| Model | Telco Accuracy | Telco F1 |
|---|---|---|
| Logistic Regression | ~80% | ~0.58 |
| Random Forest (tuned) | ~82% | ~0.62 |
| Gradient Boosting | ~81% | ~0.61 |

> Full results and confusion matrices are in the notebook.

### Key Insights
- **Stratified sampling** is critical for imbalanced churn data (~26% churn rate)
- **GridSearchCV with F1** as scoring metric outperforms accuracy-based tuning for imbalanced classes
- **Random Forest with depth limiting** reduces overfitting significantly on German Credit
- **TotalCharges** and **tenure** are consistently top predictors of churn across models

---

## 📂 Project Structure

```
📁 ml-model-optimization-churn/
│
├── 📓 Phase3_Group11.ipynb         ← Main analysis notebook
├── 📄 phase3_report.docx           ← Full project report
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
└── 📁 data/
    ├── WA_Fn-UseC_-Telco-Customer-Churn.csv    
    ├── credit_default.xls                       
    └── german.data                            
```

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/Tharun3052/ml-model-optimization-churn.git
cd ml-model-optimization-churn
pip install -r requirements.txt
```

### 2. Add Telco dataset and other two data sets 

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `/data/`.

### 3. Run notebook

```bash
jupyter notebook Phase3_Group11.ipynb
```

---

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
xlrd>=2.0.1
```

---

## 👥 Team

| Member | Responsibility |
|---|---|
| Tharun Reddy Marreddy | Model development & hyperparameter tuning |
| Bré Anna Kotary | Data preprocessing |
| Srinija Chowdary Garapati |Feature Engineering & EDA |
| Akshaya Paila | Descriptive Statistics | Visualizations|
**Course:** DTSC 5082 — Seminar in Research & Research Methods | University of North Texas

---

<div align="center">
Made with ❤️ | <a href="https://github.com/Tharun3052">Tharun Reddy</a>
</div>
