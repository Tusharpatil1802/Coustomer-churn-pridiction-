# 📉 Customer Churn Prediction Pipeline

A machine learning pipeline that predicts which telecom customers are likely to churn — trained on a synthetic 5,000-customer dataset using **Random Forest** and **XGBoost**.

---

## 🚀 Quick Results

| Metric | Random Forest | XGBoost |
|---|---|---|
| ROC-AUC | 0.9558 | **0.9601** |
| Average Precision | 0.9105 | **0.9153** |
| F1 Score | **0.8348** | 0.8210 |
| CV AUC (5-fold) | 0.974 ± 0.003 | **0.982 ± 0.002** |
| Accuracy | 89% | 88% |

> XGBoost wins on AUC and stability. Random Forest wins on F1.

---

## 📁 Pipeline Overview

```
Raw Data (5,000 customers)
    ↓
1. Data Generation     — Synthetic telecom data using numpy distributions
2. Preprocessing       — Encoding + feature engineering (14 → 23 features)
3. Split + SMOTE       — 80/20 stratified split, balance minority class
4. Model Training      — Random Forest & XGBoost
5. Evaluation          — ROC-AUC, F1, cross-validation, inference demo
```

---

## 🗂️ Dataset

Since no real dataset was available, 5,000 customer records were generated synthetically. Key stats:

- **5,000** customers
- **14** raw features → **23** after preprocessing
- **31.6%** churn rate (realistic class imbalance)

### Features include:
- `tenure` — months as a customer (Exponential distribution, 1–72)
- `monthly_charges` — billed amount (Normal ~$65)
- `support_calls` — number of support contacts (Poisson)
- `contract_type` — Month-to-month (55%), One-year (25%), Two-year (20%)
- `internet_service` — Fiber optic (44%), DSL (34%), None (22%)
- And more: payment method, tech support, online security, etc.

### How churn is determined:
Each customer gets a score based on a weighted formula, passed through a sigmoid. If probability > 0.5 → labelled as churned.

```
churn_score =
  -0.05 × tenure          (longer tenure = lower risk)
  +0.008 × monthly_charges (higher bill = more likely to leave)
  +0.4 × month-to-month   (strongest churn driver — no lock-in)
  -0.3 × two-year          (locked in = low risk)
  +0.3 × fiber optic       (price-conscious users)
  +0.25 × electronic check (no autopay = less sticky)
  -0.2 × tech support      (supported customers stay)
  +0.08 × support_calls    (frustration signal)
  + noise(0, 0.3)          (realistic fuzz)
```

---

## ⚙️ Preprocessing

Raw text/categorical features are converted to numbers in three steps:

**1. Binary Encoding** — Yes/No columns → 1/0
```python
df[col] = (df[col] == 'Yes').astype(int)
# Columns: tech_support, online_security, dependents, partner, paperless_billing
```

**2. One-Hot Encoding** — Multi-class columns → separate binary columns
```python
df = pd.get_dummies(df, columns=['contract_type', 'internet_service', 'payment_method'])
# Adds 10 new columns (3 + 3 + 4)
```

**3. Feature Engineering** — Two new derived features:
- `avg_monthly_spend` = `total_charges / (tenure + 1)` — captures actual spend vs listed rate
- `tenure_bucket` — groups tenure into ranges [0–12, 13–24, 25–48, 49–72 months] since churn risk is non-linear

---

## ⚖️ Class Imbalance — SMOTE

Training data has 2,734 retained vs 1,266 churned customers (2.16:1 ratio). Without fixing this, the model can cheat by always predicting "retained" and still hit ~68% accuracy.

**SMOTE** (Synthetic Minority Over-sampling Technique) fixes this:
1. Picks a churned customer
2. Finds its K nearest churned neighbours
3. Creates synthetic new points by interpolating between them
4. Repeats until classes are balanced: **2,734 vs 2,734**

> ⚠️ SMOTE is applied **only to the training set**. The test set is never touched — applying it before splitting would leak data into evaluation.

---

## 🤖 Models

### Random Forest
- Builds 300 independent decision trees on random data/feature subsets
- Averages all tree votes → robust against overfitting

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced'
)
```

### XGBoost
- Builds trees **sequentially** — each tree corrects errors from the previous one
- Uses gradient descent on the loss function
- Subsampling and column sampling add regularisation

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
```

> Both models use `StandardScaler` before training so high-range features (like `total_charges`) don't overpower binary features (like `senior_citizen`).

---

## 📊 Evaluation Metrics Explained

| Metric | What it measures |
|---|---|
| **ROC-AUC** | How well the model separates churners vs non-churners (0.5 = random, 1.0 = perfect) |
| **Average Precision** | Area under Precision-Recall curve — better than AUC when classes are imbalanced |
| **F1 Score** | Balance between precision (correct churn flags) and recall (churners caught) |
| **CV AUC (5-fold)** | Model re-trained 5 times on different data splits — confirms results aren't from a lucky split |

---

## 🔍 Inference Demo

A worst-case customer profile was tested:
- Tenure: 3 months
- Monthly charges: $95
- Month-to-month contract
- Fiber optic internet
- Electronic check payment
- Senior citizen
- 4 support calls

```
Random Forest → 99.25% churn probability  ⚠️ HIGH RISK
XGBoost       → 99.99% churn probability  ⚠️ HIGH RISK
```

Both models correctly flag this as extreme churn risk — every major risk factor is triggered simultaneously.

---

## 🏆 Top Churn Drivers (Feature Importance)

Based on Random Forest feature importances:

1. `tenure` — 0.19
2. `monthly_charges` — 0.16
3. `total_charges` — 0.14
4. `avg_monthly_spend` — 0.11
5. `support_calls` — 0.07
6. `contract_type_MTM` — 0.06

---

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** — Random Forest, preprocessing, evaluation
- **XGBoost** — Gradient boosting classifier
- **imbalanced-learn** — SMOTE
- **NumPy / Pandas** — Data generation and manipulation
- **Matplotlib / Seaborn** — Visualisation

---

## 📦 Installation

```bash
pip install scikit-learn xgboost imbalanced-learn numpy pandas matplotlib seaborn
```

---

## 📄 License

MIT
