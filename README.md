# 🏦 FinSense — Personal Loan Eligibility Predictor

> An end-to-end Data Science project: synthetic data generation → EDA → model comparison → deployment-ready Flask app with explainable AI output.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Accuracy](https://img.shields.io/badge/CV%20Accuracy-94.8%25-green) ![Model](https://img.shields.io/badge/Model-Gradient%20Boosting-cyan) ![Domain](https://img.shields.io/badge/Domain-FinTech%20%7C%20Credit%20Risk-orange)

---

## 📌 Project Objective

Build an AI-powered **personal loan eligibility predictor** that:
- Predicts approval / rejection with **94.8% accuracy**
- Explains *why* using feature importance (SHAP-style breakdown)
- Suggests max eligible loan amount, interest rate, and estimated EMI
- Gives rejected applicants **actionable tips** to improve their chances

---

## 📁 Project Structure

```
personal_loan_app/
├── app.py                            ← Flask web app (REST API + UI)
├── retrain.py                        ← Retrain model on new data
├── pl_model.pkl                      ← Trained GradientBoostingClassifier
├── pl_features.pkl                   ← Feature column order
├── pl_encoders.pkl                   ← Label encoding maps
├── personal_loan_dataset.csv         ← Synthetic training dataset (6,000 rows)
├── Personal_Loan_EDA_Analysis.ipynb  ← Full EDA + model comparison notebook
├── chart_eda_overview.png            ← EDA overview (4 panels)
├── chart_correlation.png             ← Correlation heatmap
├── chart_model_comparison.png        ← Model metrics + ROC curves
├── chart_feature_importance.png      ← Feature importance chart
├── chart_shap_waterfall.png          ← Individual prediction explanation
├── model_comparison.csv              ← Model metrics table (CSV)
├── requirements.txt
└── templates/
    └── index.html                    ← 4-step guided form UI
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Synthetic — generated from real Indian lending rules |
| Size | 6,000 applicants |
| Approval Rate | ~59.7% |
| Features | 16 input features |
| Target | `Loan_Approved` (1 = Approved, 0 = Rejected) |

---

## 🔍 Key EDA Insights

### 1. CIBIL Score — The #1 Decision Factor (44% of model weight)

| CIBIL Band | Approval Rate |
|---|---|
| 750–900 (Excellent) | **90.3%** |
| 700–749 (Good) | ~75% |
| 650–699 (Average) | ~45% |
| 550–649 (Fair) | ~19% |
| 300–549 (Poor) | **3.3%** |

> A score jump from 580 → 760 changes approval odds by ~87 percentage points.

### 2. Debt-to-Income Ratio — The #2 Factor (27.6%)

| DTI | Approval Rate |
|---|---|
| < 30% | **89.8%** |
| 30–50% | ~73% |
| 50–65% | ~38% |
| > 65% | **~11%** |

> Clearing one loan to drop DTI from 65% to 40% can flip a rejection to approval.

### 3. Income Impact

| Monthly Income | Approval Rate |
|---|---|
| ≥ ₹1,00,000 | **88.6%** |
| ₹50K–₹1L | ~72% |
| < ₹25,000 | ~21% |

### 4. Employer Category

| Employer | Approval Rate |
|---|---|
| Government / PSU | **67.5%** |
| MNC | 64.2% |
| Private SME | 60.1% |
| Startup | 57.5% |

### 5. The "Golden Zone"
Applicants with **CIBIL > 700 AND DTI < 50%** are approved **~92% of the time**.

---

## 🤖 Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Acc |
|---|---|---|---|---|---|---|
| Logistic Regression | 81.4% | 82.6% | 87.5% | 84.9% | 0.886 | 81.7% |
| Random Forest | 92.2% | 91.8% | 94.7% | 93.2% | 0.980 | 93.3% |
| **Gradient Boosting** | **94.3%** | **94.3%** | **96.1%** | **95.1%** | **0.990** | **94.8%** |

### Why Gradient Boosting?

1. **Best across all 5 metrics** — not just accuracy
2. **ROC-AUC 0.990** — near-perfect class separation
3. **Sequential boosting** — each tree corrects errors of previous trees, ideal for tabular financial data
4. **Captures non-linear interactions** — CIBIL × DTI combined effect is key
5. **Low variance** — CV std 0.006, generalises well to unseen data
6. Logistic Regression underperforms because approval boundaries are non-linear
7. Random Forest is close but GBM extracts more signal via boosting

---

## 🎯 App Output — Full Breakdown

| Output | Example |
|---|---|
| Decision | Approved ✅ / Rejected ❌ |
| Confidence | 94.2% |
| Max Eligible Loan | ₹8,40,000 |
| Interest Rate | 11.5% p.a. |
| Monthly EMI | ₹9,893 |
| ✅ Strengths | CIBIL 762, MNC employer, low DTI |
| ⚠️ Concerns | High DTI, low experience |
| 💡 Tips | "Clear existing EMI to reduce DTI below 50%" |

---

## 🚀 Quick Start

```bash
cd personal_loan_app
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Run the EDA Notebook
```bash
pip install jupyter
jupyter notebook Personal_Loan_EDA_Analysis.ipynb
```

### Retrain Model
```bash
python retrain.py                      # fresh synthetic data
python retrain.py --data my_data.csv  # your own dataset
```

---

## 🌐 Deploy Free on Render

1. Push to GitHub
2. [render.com](https://render.com) → New Web Service → connect repo
3. Build: `pip install -r requirements.txt` | Start: `gunicorn app:app`
4. Live in ~5 minutes ✅

---

## 🔌 REST API Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age":"28","Gender":"Male","Marital_Status":"Single","Dependents":"0",
    "Education":"Master","Employment_Type":"Salaried","Employer_Category":"MNC",
    "Work_Experience_Yrs":"4","City_Tier":"Tier 1",
    "Net_Monthly_Income":"75000","Monthly_Expenses":"28000","Existing_EMI":"5000",
    "CIBIL_Score":"740","Loan_Amount_Requested":"300000","Loan_Tenure_Months":"36"
  }'
```

---

## 📈 Skills Demonstrated

- ✅ Synthetic dataset generation with domain-realistic business logic
- ✅ Exploratory Data Analysis — charts that tell a business story
- ✅ Feature engineering — DTI ratio, CIBIL bands, categorical encoding
- ✅ Model comparison — Logistic Regression vs Random Forest vs Gradient Boosting
- ✅ Rigorous evaluation — accuracy, precision, recall, F1, ROC-AUC, 5-fold CV
- ✅ Explainable AI — feature importance + per-prediction breakdown
- ✅ REST API with JSON input/output
- ✅ Production Flask app with input validation and error handling
- ✅ One-command retraining pipeline
- ✅ Deployment-ready (Gunicorn + Render)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn (GradientBoostingClassifier) |
| Web | Flask + Gunicorn |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Frontend | Vanilla HTML / CSS / JS |
