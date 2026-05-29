# BNPL Credit Risk Assessment with Hybrid Decision System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

##  Project Overview

This project develops a machine learning solution for assessing credit risk in Buy Now, Pay Later (BNPL) transactions. Using a real-world fintech dataset, we build a CatBoost classifier to predict customer default probability.

The system implements a hybrid decision framework that automatically approves low-risk applicants, rejects high-risk ones, and flags borderline cases for manual review — balancing operational efficiency with risk control.

---

##  Key Features

-  End-to-end data analysis and feature engineering
-  CatBoost model with hyperparameter tuning (AUC = 0.788)
-  Hybrid rule-based + ML decision system
-  Streamlit interactive dashboard for deployment
-  Business-friendly metrics (automation rate, approval accuracy, risk exposure)

---

##  Dataset

**Source:** Buy Now Pay Later Fintech ML Dataset

The dataset contains **10,345 records** of BNPL transactions with **17 features**, including:

| Feature | Description |
|---|---|
| `credit_score` | Customer’s credit score |
| `debt_to_income_ratio` | Debt-to-income ratio |
| `missed_payments` | Number of missed BNPL payments |
| `repayment_delay_days` | Days delayed in repayment |
| `risk_score` | Internal risk score (0-100) |
| `bnpl_installments` | Number of installments chosen |
| `monthly_income`, `age`, `purchase_amount` | Demographics and transaction info |
| `default_flag` | Target variable (1 = default, 0 = non-default) |

---

##  Exploratory Data Analysis (EDA)

### Key Insights

- Class imbalance: **39% default rate**, **61% non-default**
- Strongest positive correlations with default:
  - `risk_score` (**0.40**)
  - `repayment_delay_days` (**0.28**)
  - `missed_payments` (**0.27**)
- Strongest negative correlation:
  - `credit_score` (**-0.32**)

### Default Rate Increases Sharply With

- Repayment delays > 15 days (**32% → 73%**)
- Missed payments ≥ 3 (**exceeds 80%**)
- Debt-to-income ratio > 0.36 (**above 48%**)

### Additional Findings

- No significant difference in default rates across:
  - Product categories
  - Locations
  - BNPL installment counts

 Visualizations such as **correlation heatmaps, box plots, and KDE plots** are included in the notebook.

---

##  Feature Engineering

We created custom features to capture deeper risk patterns and improve model interpretability.

| Feature | Formula |
|---|---|
| `payment_compliance` | `1 – (missed_payments / (installments + 1))` |
| `monthly_payment` | `purchase_amount / installments` |
| `debt_burden` | `monthly_payment / (monthly_income + 1)` |
| `risk_adjusted_credit` | `credit_score × (1 – risk_score/100)` |
| `high_risk_customer` | `(risk_score > 70 OR credit_score < 580) → 1` |
| `severe_delinquent` | `(repayment_delay_days > 30) → 1` |

These engineered features improved both model discrimination and business interpretability.

---

##  Modeling

We used **CatBoost** for its:

- Native handling of categorical features
- Strong performance on tabular data
- Robustness to overfitting

###  Hyperparameter Tuning

Randomized Search with **5-fold cross-validation** optimized the following parameters:

```python
best_params = {
    'iterations': 100,
    'learning_rate': 0.15,
    'depth': 7,
    'l2_leaf_reg': 3,
    'random_state': 42
}
```

---

##  Model Performance

| Metric | Value |
|---|---|
| Cross-validation AUC | 0.7627 (±0.0072) |
| Test AUC | 0.7880 |
| Test F1 (default class) | 0.568 |
| Accuracy (0.5 threshold) | 74% |

### Classification Report (Threshold = 0.5)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Non-Default | 0.72 | 0.93 | 0.81 |
| Default | 0.80 | 0.44 | 0.57 |

---

##  Hybrid Decision System

Instead of using a single prediction threshold, the system applies a **three-zone hybrid decision strategy**.

| Zone | Probability Range | Action |
|---|---|---|
|  Approve | `< 0.30` | Automatic approval |
|  Review | `0.30 – 0.60` | Manual underwriting |
|  Reject | `> 0.60` | Automatic rejection |

---

##  Business Impact

| Metric | Value |
|---|---|
| Automation rate | 43.0% |
| Auto-approval accuracy | 93.1% |
| Auto-rejection accuracy | 83.3% |
| Defaulters flagged for review | 444 out of 808 |

### Breakdown

- 23.8% automatically approved
- 19.1% automatically rejected
- Only 34 defaulters were auto-approved
- 66 good customers were auto-rejected

This hybrid framework significantly reduces manual underwriting workload while maintaining strong risk control.

---

##  How to Run Locally

### Prerequisites

- Python 3.9+
- pip

---

###  Clone the Repository

```bash
git clone https://github.com/yourusername/bnpl-credit-risk.git
cd bnpl-credit-risk
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Run the Jupyter Notebook (Optional)

```bash
jupyter notebook BNPL_Credit_Risk.ipynb
```

###  Run the Streamlit App

```bash
streamlit run app.py
```

The application loads the saved model (`bnpl_hybrid_model.pkl`) and provides real-time customer risk decisions:

-  Approve
-  Review
-  Reject

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Matplotlib
- Seaborn
- Streamlit

---

##  Future Improvements

- Deploy using Docker + Cloud services
- Add SHAP explainability dashboard
- Integrate real-time API scoring
- Implement drift monitoring and retraining pipeline
- Add advanced ensemble models

---

## 📄 License

This project is licensed under the MIT License.
