"""
Customer Churn Prediction - Full ML Pipeline
=============================================
Models  : Random Forest + XGBoost
Data    : Synthetic (telecom-style)
Handles : Class imbalance (SMOTE), preprocessing, evaluation, feature importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except (ImportError, Exception) as e:
    print(f"Warning: XGBoost could not be loaded ({e}). Continuing without XGBoost.")
    XGBClassifier = None
from imblearn.over_sampling import SMOTE

import os
os.makedirs("./outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
def generate_churn_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    n = n_samples

    tenure          = np.random.exponential(scale=30, size=n).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 25, size=n).clip(20, 150).round(2)
    total_charges   = (tenure * monthly_charges * np.random.uniform(0.85, 1.05, size=n)).round(2)
    num_products    = np.random.choice([1, 2, 3, 4], size=n, p=[0.3, 0.35, 0.25, 0.1])
    contract_type   = np.random.choice(["Month-to-month", "One year", "Two year"],
                                        size=n, p=[0.55, 0.25, 0.20])
    internet_service= np.random.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])
    payment_method  = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n, p=[0.34, 0.23, 0.22, 0.21]
    )
    tech_support    = np.random.choice(["Yes", "No"], size=n, p=[0.29, 0.71])
    online_security = np.random.choice(["Yes", "No"], size=n, p=[0.28, 0.72])
    senior_citizen  = np.random.choice([0, 1], size=n, p=[0.84, 0.16])
    dependents      = np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    partner         = np.random.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    paperless       = np.random.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    support_calls   = np.random.poisson(lam=1.5, size=n).clip(0, 10)

    # Churn probability driven by real-world logic
    churn_score = (
        -0.05 * tenure
        + 0.008 * monthly_charges
        + 0.4  * (contract_type == "Month-to-month")
        - 0.3  * (contract_type == "Two year")
        + 0.3  * (internet_service == "Fiber optic")
        + 0.25 * (payment_method == "Electronic check")
        - 0.2  * (tech_support == "Yes")
        - 0.2  * (online_security == "Yes")
        + 0.15 * senior_citizen
        - 0.1  * (dependents == "Yes")
        + 0.08 * support_calls
        - 0.15 * num_products
        + np.random.normal(0, 0.3, size=n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn      = (churn_prob > 0.5).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_products": num_products,
        "support_calls": support_calls,
        "senior_citizen": senior_citizen,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "tech_support": tech_support,
        "online_security": online_security,
        "dependents": dependents,
        "partner": partner,
        "paperless_billing": paperless,
        "churn": churn
    })
    return df

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    # Encode binary/ordinal categoricals
    binary_cols = ["tech_support", "online_security", "dependents", "partner", "paperless_billing"]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)

    # One-hot encode multi-class categoricals
    df = pd.get_dummies(df, columns=["contract_type", "internet_service", "payment_method"], drop_first=False)

    # Feature engineering
    df["avg_monthly_spend"] = df["total_charges"] / (df["tenure"] + 1)
    df["tenure_bucket"]     = pd.cut(df["tenure"], bins=[0,12,24,48,72], labels=[0,1,2,3]).astype(int)

    X = df.drop("churn", axis=1)
    y = df["churn"]
    return X, y

# ─────────────────────────────────────────────
# 3. TRAIN / EVALUATE
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # SMOTE on training set only
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_tr_sc, y_train)
    print(f"\n[SMOTE] Training set before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[SMOTE] Training set after : {dict(zip(*np.unique(y_res, return_counts=True)))}")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    }
    
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1, use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )

    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
        model.fit(X_res, y_res)

        y_pred      = model.predict(X_te_sc)
        y_prob      = model.predict_proba(X_te_sc)[:, 1]
        roc_auc     = roc_auc_score(y_test, y_prob)
        avg_prec    = average_precision_score(y_test, y_prob)
        f1          = f1_score(y_test, y_pred)

        # CV on resampled data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring="roc_auc", n_jobs=-1)

        print(f"ROC-AUC     : {roc_auc:.4f}")
        print(f"Avg Prec    : {avg_prec:.4f}")
        print(f"F1 Score    : {f1:.4f}")
        print(f"CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        # Feature importances
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        feat_imp_df = feat_imp_df.sort_values("importance", ascending=False).head(15)

        results[name] = {
            "model": model, "y_pred": y_pred, "y_prob": y_prob,
            "roc_auc": roc_auc, "avg_prec": avg_prec, "f1": f1,
            "cv_scores": cv_scores, "feat_imp": feat_imp_df,
            "cm": confusion_matrix(y_test, y_pred)
        }

    return results, scaler

# ─────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────
PALETTE = {"Random Forest": "#2563EB", "XGBoost": "#16A34A"}
BG      = "#0F172A"
CARD    = "#1E293B"
TEXT    = "#F1F5F9"
MUTED   = "#94A3B8"

def styled_ax(ax, title=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

def plot_dashboard(results, y_test, df):
    fig = plt.figure(figsize=(22, 18), facecolor=BG)
    fig.suptitle("Customer Churn Prediction — Model Evaluation Dashboard",
                 color=TEXT, fontsize=18, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

    # ── ROC Curves ──
    ax_roc = fig.add_subplot(gs[0, :2])
    styled_ax(ax_roc, "ROC Curve")
    ax_roc.plot([0,1],[0,1], "--", color="#475569", lw=1)
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax_roc.plot(fpr, tpr, color=PALETTE[name], lw=2,
                    label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(facecolor=CARD, edgecolor="#334155", labelcolor=TEXT, fontsize=9)

    # ── Precision-Recall ──
    ax_pr = fig.add_subplot(gs[0, 2:])
    styled_ax(ax_pr, "Precision-Recall Curve")
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax_pr.plot(rec, prec, color=PALETTE[name], lw=2,
                   label=f"{name} (AP={res['avg_prec']:.3f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(facecolor=CARD, edgecolor="#334155", labelcolor=TEXT, fontsize=9)

    # ── Confusion Matrices ──
    for i, (name, res) in enumerate(results.items()):
        ax_cm = fig.add_subplot(gs[1, i*2:(i+1)*2])
        styled_ax(ax_cm, f"Confusion Matrix — {name}")
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Retained","Churned"],
                    yticklabels=["Retained","Churned"],
                    ax=ax_cm, cbar=False,
                    annot_kws={"color": TEXT, "fontsize": 12, "fontweight": "bold"})
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")

    # ── Feature Importance ──
    for i, (name, res) in enumerate(results.items()):
        ax_fi = fig.add_subplot(gs[2, i*2:(i+1)*2])
        styled_ax(ax_fi, f"Top 15 Feature Importances — {name}")
        fi = res["feat_imp"]
        bars = ax_fi.barh(fi["feature"][::-1], fi["importance"][::-1],
                          color=PALETTE[name], alpha=0.85)
        for bar in bars:
            w = bar.get_width()
            ax_fi.text(w + 0.001, bar.get_y() + bar.get_height()/2,
                       f"{w:.3f}", va="center", ha="left", color=MUTED, fontsize=8)

    plt.savefig("./outputs/churn_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor=BG)
    print("\n[✓] Dashboard saved → outputs/churn_dashboard.png")

def plot_metrics_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
    fig.suptitle("Model Comparison — Key Metrics", color=TEXT, fontsize=14, fontweight="bold")

    metrics = ["roc_auc", "avg_prec", "f1"]
    labels  = ["ROC-AUC", "Avg Precision", "F1 Score"]
    names   = list(results.keys())
    colors  = [PALETTE[n] for n in names]

    for ax, metric, label in zip(axes, metrics, labels):
        styled_ax(ax, label)
        vals = [results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, width=0.45, alpha=0.9)
        ax.set_ylim(0, 1.0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", color=TEXT, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=10, colors=TEXT)

    plt.tight_layout()
    plt.savefig("./outputs/metrics_comparison.png",
                dpi=150, bbox_inches="tight", facecolor=BG)
    print("[✓] Metrics comparison saved → outputs/metrics_comparison.png")

# ─────────────────────────────────────────────
# 5. INFERENCE DEMO
# ─────────────────────────────────────────────
def predict_single_customer(model, scaler, feature_names, customer_dict):
    """Predict churn probability for a single customer."""
    row = pd.DataFrame([customer_dict]).reindex(columns=feature_names, fill_value=0)
    row_sc  = scaler.transform(row)
    prob    = model.predict_proba(row_sc)[0][1]
    label   = "CHURN RISK 🔴" if prob > 0.5 else "RETAINED  🟢"
    print(f"\n  Customer Churn Probability : {prob:.2%}  →  {label}")
    return prob

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("   CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 60)

    # 1. Generate data
    print("\n[1/5] Generating synthetic dataset...")
    df = generate_churn_data(n_samples=5000)
    df.to_csv("./outputs/synthetic_churn_data.csv", index=False)
    print(f"      Rows: {len(df)} | Churn rate: {df['churn'].mean():.1%}")

    # 2. Preprocess
    print("\n[2/5] Preprocessing...")
    X, y = preprocess(df)
    feature_names = list(X.columns)
    print(f"      Features: {len(feature_names)}")

    # 3. Split
    print("\n[3/5] Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train & evaluate
    print("\n[4/5] Training models...")
    results, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

    # 5. Visualise
    print("\n[5/5] Generating visualisations...")
    plot_dashboard(results, y_test, df)
    plot_metrics_comparison(results)

    # 6. Inference demo
    print("\n" + "─"*60)
    print("INFERENCE DEMO — Single Customer Prediction")
    print("─"*60)
    sample_customer = {
        "tenure": 3, "monthly_charges": 95.0, "total_charges": 285.0,
        "num_products": 1, "support_calls": 4, "senior_citizen": 1,
        "tech_support": 0, "online_security": 0, "dependents": 0,
        "partner": 0, "paperless_billing": 1,
        "avg_monthly_spend": 95.0, "tenure_bucket": 0,
        "contract_type_Month-to-month": 1, "contract_type_One year": 0, "contract_type_Two year": 0,
        "internet_service_DSL": 0, "internet_service_Fiber optic": 1, "internet_service_No": 0,
        "payment_method_Electronic check": 1, "payment_method_Mailed check": 0,
        "payment_method_Bank transfer": 0, "payment_method_Credit card": 0,
    }
    print("\n  Profile: Tenure=3mo, ₹95/mo, Month-to-month, Fiber, Electronic check, Senior")
    for name, res in results.items():
        print(f"\n  [{name}]")
        predict_single_customer(res["model"], scaler, feature_names, sample_customer)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Outputs saved to: outputs/")
    print("=" * 60)
