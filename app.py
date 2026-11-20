import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

st.set_page_config(
    page_title="Credit Card Default — Feature & Cost Comparison",
    layout="wide",
)

RANDOM_STATE = 42
APPROACHES = ["original", "average", "feature_engineered"]
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMOUNT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def make_features(data: pd.DataFrame, approach: str) -> pd.DataFrame:
    result = data.copy()

    if approach == "original":
        return result

    if approach == "average":
        groups = [
            (PAY_COLS, "avg_pay_status"),
            (BILL_COLS, "avg_bill_amount"),
            (AMOUNT_COLS, "avg_payment_amount"),
        ]
        for cols, new_col in groups:
            available = [c for c in cols if c in result.columns]
            if available:
                result[new_col] = result[available].mean(axis=1)
                result = result.drop(columns=available)
        return result

    if approach == "feature_engineered":
        pay = [c for c in PAY_COLS if c in result.columns]
        bills = [c for c in BILL_COLS if c in result.columns]
        payments = [c for c in AMOUNT_COLS if c in result.columns]

        if pay:
            result["avg_pay_status"] = result[pay].mean(axis=1)
            result["worst_pay_status"] = result[pay].max(axis=1)
            result["months_late"] = (result[pay] > 0).sum(axis=1)
            result["recent_pay_status"] = result[pay[0]]
        if bills:
            result["avg_bill_amount"] = result[bills].mean(axis=1)
            result["bill_amount_change"] = result[bills[0]] - result[bills[-1]]
        if payments:
            result["avg_payment_amount"] = result[payments].mean(axis=1)
            result["total_payment_amount"] = result[payments].sum(axis=1)
        if bills and payments:
            result["overall_payment_bill_ratio"] = (
                result["avg_payment_amount"] / (result["avg_bill_amount"].abs() + 1)
            )
        if "LIMIT_BAL" in result.columns and bills:
            result["average_credit_utilization"] = (
                result["avg_bill_amount"] / (result["LIMIT_BAL"].abs() + 1)
            )
        return result

    raise ValueError(f"Unknown feature approach: {approach}")


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            ),
        ),
        "SVM": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced", max_iter=10000, random_state=RANDOM_STATE
                ),
                method="sigmoid",
                cv=3,
            ),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=8,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced_subsample",
                min_samples_leaf=5,
                n_jobs=2,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def compute_business_cost(y_true, y_pred, cost_fp: float, cost_fn: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Total cost": float(fp * cost_fp + fn * cost_fn),
    }


def get_metrics(y_true, probability, threshold: float, cost_fp: float, cost_fn: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "PR-AUC": average_precision_score(y_true, probability),
        **compute_business_cost(y_true, prediction, cost_fp, cost_fn),
    }


@st.cache_data(show_spinner=False)
def load_and_split_data(file_path_or_bytes):
    df = pd.read_excel(file_path_or_bytes, skiprows=1)
    df = df.rename(
        columns={
            "PAY_0": "PAY_1",
            "default payment next month": "default_payment",
        }
    )
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.dropna(subset=["default_payment"]).copy()
    df["default_payment"] = df["default_payment"].astype(int)

    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    target = "default_payment"
    X = df.drop(columns=target)
    y = df[target]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )
    return df, X_train, X_val, X_test, y_train, y_val, y_test


@st.cache_data(show_spinner=True)
def run_experiments(X_train, X_val, X_test, y_train, y_val, y_test, cost_fp: float, cost_fn: float):
    thresholds = np.arange(0.01, 0.991, 0.01)
    validation_rows = []
    validation_cost_curves = {}
    val_probs_store = {}
    test_probs_store = {}

    for approach in APPROACHES:
        Xtr = make_features(X_train, approach)
        Xv = make_features(X_val, approach)
        Xte = make_features(X_test, approach)

        for model_name, model in build_models().items():
            key = (approach, model_name)
            model.fit(Xtr, y_train)

            val_prob = model.predict_proba(Xv)[:, 1]
            test_prob = model.predict_proba(Xte)[:, 1]
            val_probs_store[key] = val_prob
            test_probs_store[key] = test_prob

            at_half = get_metrics(y_val, val_prob, 0.50, cost_fp, cost_fn)
            costs = [
                compute_business_cost(y_val, (val_prob >= t).astype(int), cost_fp, cost_fn)["Total cost"]
                for t in thresholds
            ]
            best_idx = int(np.argmin(costs))
            best_threshold = float(thresholds[best_idx])
            best_metrics = get_metrics(y_val, val_prob, best_threshold, cost_fp, cost_fn)

            validation_rows.append(
                {
                    "Feature approach": approach,
                    "Model": model_name,
                    "Features": Xtr.shape[1],
                    "Accuracy at 0.50": at_half["Accuracy"],
                    "Precision at 0.50": at_half["Precision"],
                    "Recall at 0.50": at_half["Recall"],
                    "F1 at 0.50": at_half["F1"],
                    "ROC-AUC": at_half["ROC-AUC"],
                    "PR-AUC": at_half["PR-AUC"],
                    "Best threshold (cost)": best_threshold,
                    "Validation cost (best threshold)": best_metrics["Total cost"],
                    "Accuracy at best threshold": best_metrics["Accuracy"],
                    "Precision at best threshold": best_metrics["Precision"],
                    "Recall at best threshold": best_metrics["Recall"],
                }
            )
            validation_cost_curves[key] = costs

    return (
        pd.DataFrame(validation_rows),
        thresholds,
        validation_cost_curves,
        val_probs_store,
        test_probs_store,
    )


def optimize_threshold_for_costs(y_true, prob, thresholds, cost_fp: float, cost_fn: float):
    costs = [
        compute_business_cost(y_true, (prob >= t).astype(int), cost_fp, cost_fn)["Total cost"]
        for t in thresholds
    ]
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), costs


# --- Sidebar Controls ---
st.sidebar.header("Dataset & Cost Settings")
uploaded_file = st.sidebar.file_uploader("Upload project.xls (optional)", type=["xls", "xlsx"])
data_source = uploaded_file if uploaded_file is not None else "project.xls"

cost_fp = st.sidebar.number_input("False Positive Cost (FP)", min_value=0, value=1000, step=100)
cost_fn = st.sidebar.number_input("False Negative Cost (FN)", min_value=0, value=5000, step=100)

st.title("Credit Card Default Prediction — Feature & Business Cost Comparison")
st.markdown(
    "Compares **Original**, **Average Aggregation**, and **Feature Engineering** across "
    "**Logistic Regression**, **SVM**, **Decision Tree**, and **Random Forest** using a shared "
    "60/20/20 stratified train/validation/test split. Thresholds are selected on validation data only."
)
st.caption(
    "Illustrative project assumptions: flagged accounts should undergo human review rather than automatic rejection."
)

try:
    df, X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(data_source)
except Exception as exc:
    st.error(f"Could not load dataset from `{data_source}`: {exc}")
    st.stop()

# Top summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset Rows", f"{len(df):,}")
c2.metric("Default Rate", f"{y_train.mean():.2%}")
c3.metric("Split (Train / Val / Test)", f"{len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
c4.metric("Active Cost Ratio (FN:FP)", f"{cost_fn / cost_fp:.1f}:1" if cost_fp > 0 else "N/A")

# Feature approach counts
feat_summary = pd.DataFrame(
    [{"Feature approach": a, "Input features": make_features(X_train, a).shape[1]} for a in APPROACHES]
)

with st.expander("Dataset Preview & Feature Counts per Approach", expanded=False):
    st.dataframe(feat_summary, hide_index=True)
    st.dataframe(df.head(10))

validation_results, thresholds, validation_cost_curves, val_probs_store, test_probs_store = run_experiments(
    X_train, X_val, X_test, y_train, y_val, y_test, float(cost_fp), float(cost_fn)
)

# --- Section 6 & 7: Validation Comparison & Best per Approach ---
st.header("1. Validation Comparison Across All 12 Combinations")
sorted_val = validation_results.sort_values("Validation cost (best threshold)").reset_index(drop=True)
st.dataframe(sorted_val.round(4))

st.header("2. Best Model per Feature Approach (by Validation Cost)")
best_per_approach = (
    validation_results.sort_values("Validation cost (best threshold)")
    .groupby("Feature approach", as_index=False)
    .first()
)
st.dataframe(
    best_per_approach[
        [
            "Feature approach",
            "Model",
            "Features",
            "Accuracy at 0.50",
            "Precision at 0.50",
            "Recall at 0.50",
            "F1 at 0.50",
            "ROC-AUC",
            "PR-AUC",
            "Best threshold (cost)",
            "Validation cost (best threshold)",
        ]
    ].round(4),
    hide_index=True,
)

fig_curve, ax_curve = plt.subplots(figsize=(10, 4.5))
for _, row in best_per_approach.iterrows():
    key = (row["Feature approach"], row["Model"])
    ax_curve.plot(thresholds, validation_cost_curves[key], label=f"{key[0]} — {key[1]}")
ax_curve.set_xlabel("Decision threshold")
ax_curve.set_ylabel("Validation business cost")
ax_curve.set_title("Validation cost by threshold — best model per feature approach")
ax_curve.legend()
fig_curve.tight_layout()
st.pyplot(fig_curve)

# --- Section 8 & 9: Final Model Selection & Test Evaluation ---
st.header("3. Final Model Selection & Test Set Evaluation")
best_row = validation_results.loc[validation_results["Validation cost (best threshold)"].idxmin()]
best_approach = best_row["Feature approach"]
best_model_name = best_row["Model"]
best_threshold = float(best_row["Best threshold (cost)"])
best_test_prob = test_probs_store[(best_approach, best_model_name)]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Approach", best_approach)
m2.metric("Selected Classifier", best_model_name)
m3.metric("Selected Threshold", f"{best_threshold:.2f}")
m4.metric("Min Validation Cost", f"{best_row['Validation cost (best threshold)']:,.0f}")

test_rows = []
for label, thr in [
    ("Standard threshold (0.50)", 0.50),
    ("Cost-optimized threshold", best_threshold),
]:
    metrics = get_metrics(y_test, best_test_prob, thr, float(cost_fp), float(cost_fn))
    test_rows.append({"Decision rule": label, "Threshold": thr, **metrics})

test_results_df = pd.DataFrame(test_rows).set_index("Decision rule")
st.dataframe(test_results_df.round(4))

col_cm, col_info = st.columns([1, 1])
with col_cm:
    final_pred = (best_test_prob >= best_threshold).astype(int)
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        final_pred,
        display_labels=["No default", "Default"],
        values_format="d",
        ax=ax_cm,
    )
    ax_cm.set_title(f"Test confusion matrix — {best_approach} / {best_model_name}")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)

# --- Section 10: Compare 5:1 and 3:1 Cost Ratios ---
st.header("4. Business Cost Ratio Comparison (5:1 vs. 3:1)")
best_val_prob = val_probs_store[(best_approach, best_model_name)]

ratio_rows = []
for ratio_label, fp_c, fn_c in [("5:1", 1000.0, 5000.0), ("3:1", 1000.0, 3000.0)]:
    val_thr, _ = optimize_threshold_for_costs(y_val, best_val_prob, thresholds, fp_c, fn_c)
    t_metrics = get_metrics(y_test, best_test_prob, val_thr, fp_c, fn_c)
    ratio_rows.append(
        {
            "Cost ratio FN:FP": ratio_label,
            "FP cost": int(fp_c),
            "FN cost": int(fn_c),
            "Validation threshold": val_thr,
            "Test accuracy": t_metrics["Accuracy"],
            "Test precision": t_metrics["Precision"],
            "Test recall": t_metrics["Recall"],
            "Test F1": t_metrics["F1"],
            "Test ROC-AUC": t_metrics["ROC-AUC"],
            "Test PR-AUC": t_metrics["PR-AUC"],
            "Test FP": t_metrics["FP"],
            "Test FN": t_metrics["FN"],
            "Test total cost": t_metrics["Total cost"],
        }
    )

st.dataframe(pd.DataFrame(ratio_rows).round(4), hide_index=True)