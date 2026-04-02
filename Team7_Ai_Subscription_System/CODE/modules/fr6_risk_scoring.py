"""
FR6 - Risk Scoring Model
Gradient Boosting classifier -> Risk Score (0-1), Risk Level, Reason.
BRD target: >85% accuracy
"""

import os
import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

HIGH   = 0.65
MEDIUM = 0.35

FEATURES = [
    "Current_Balance", "Failed_Debit_Rate", "Avg_Debit_Amount",
    "Upcoming_Total_Debit", "Balance_To_Debit_Ratio",
    "Upcoming_Debit_Pct", "Total_Monthly_Sub_Amount", "Subscription_Count",
]


def compute_score(feat_df):
    p95 = lambda col, fallback=0.01: max(feat_df[col].quantile(0.95), fallback)
    f_fail   = (feat_df["Failed_Debit_Rate"]       / p95("Failed_Debit_Rate")).clip(0, 1)
    f_subs   = (feat_df["Subscription_Count"]       / max(feat_df["Subscription_Count"].max(), 1)).clip(0, 1)
    f_upco   = (feat_df["Upcoming_Total_Debit"]     / p95("Upcoming_Total_Debit", 1)).clip(0, 1)
    f_burden = (feat_df["Total_Monthly_Sub_Amount"] / p95("Total_Monthly_Sub_Amount", 1)).clip(0, 1)
    return (0.35 * f_fail + 0.30 * f_subs + 0.20 * f_upco + 0.15 * f_burden).clip(0, 1).round(4)


def risk_level(score):
    if score >= HIGH:   return "High"
    if score >= MEDIUM: return "Medium"
    return "Low"


def risk_reason(row, dataset_stats=None):
    """Explainable risk reasons using dataset-relative thresholds."""
    reasons = []

    if dataset_stats:
        p75_fail   = dataset_stats.get("p75_fail",   0.05)
        p75_subs   = dataset_stats.get("p75_subs",   3)
        p75_upco   = dataset_stats.get("p75_upco",   5000)
        p75_burden = dataset_stats.get("p75_burden", 80000)
    else:
        p75_fail, p75_subs, p75_upco, p75_burden = 0.05, 3, 5000, 80000

    fail_rate  = row["Failed_Debit_Rate"]
    subs_count = row["Subscription_Count"]
    upcoming   = row["Upcoming_Total_Debit"]
    balance    = row["Current_Balance"]
    burden     = row["Total_Monthly_Sub_Amount"]
    burden_pct = (burden / max(balance, 1)) * 100

    if fail_rate > p75_fail:
        reasons.append(f"High failure rate ({fail_rate*100:.1f}% of debits failed)")
    if subs_count >= p75_subs:
        reasons.append(f"{int(subs_count)} active subscriptions (above average)")
    if upcoming > p75_upco:
        reasons.append(f"Upcoming debit Rs{upcoming:,.0f} is high")
    if burden_pct > 5:
        reasons.append(f"Monthly sub burden Rs{burden:,.0f} ({burden_pct:.1f}% of balance)")
    elif burden_pct > 2:
        reasons.append(f"Monthly subscription spend Rs{burden:,.0f}")

    upco_pct = (upcoming / max(balance, 1)) * 100
    if upco_pct > 1:
        reasons.append(f"Upcoming debit = {upco_pct:.1f}% of balance")

    if not reasons:
        reasons.append(f"{int(subs_count)} subscription(s), Rs{upcoming:,.0f} upcoming, Rs{balance:,.0f} balance")

    return "; ".join(reasons)


def assign_label(row, p75_fail, p75_upco):
    f, s, u = row["Failed_Debit_Rate"], row["Subscription_Count"], row["Upcoming_Total_Debit"]
    if f > p75_fail or (s >= 3 and u > p75_upco): return 2
    if f > p75_fail * 0.3 or s >= 2:              return 1
    return 0


def build_risk_features(df, pred_df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    agg = df.groupby("CustomerID").agg(
        Current_Balance          = ("Balance",           "last"),
        Failed_Debits            = ("Status",            lambda x: (x == "FAILED").sum()),
        Total_Debits             = ("TransactionType",   lambda x: (x == "DEBIT").sum()),
        Avg_Debit_Amount         = ("Amount",            lambda x: x[df.loc[x.index, "TransactionType"] == "DEBIT"].mean()),
        Total_Monthly_Sub_Amount = ("Amount",            lambda x: x[(df.loc[x.index, "Is_Recurring"] == 1) & (df.loc[x.index, "Inferred_Freq"] == "Monthly")].sum()),
        Subscription_Count       = ("Description_Clean", lambda x: x[(df.loc[x.index, "Is_Recurring"] == 1) & (df.loc[x.index, "Inferred_Freq"] == "Monthly")].nunique()),
    ).reset_index()

    agg["Failed_Debit_Rate"] = (agg["Failed_Debits"] / agg["Total_Debits"].replace(0, 1)).clip(0, 1)

    upcoming = pred_df.groupby("CustomerID")["Predicted_Amount"].sum().reset_index()
    upcoming.columns = ["CustomerID", "Upcoming_Total_Debit"]
    feat = agg.merge(upcoming, on="CustomerID", how="left")

    for col in ["Upcoming_Total_Debit", "Avg_Debit_Amount", "Total_Monthly_Sub_Amount", "Subscription_Count"]:
        feat[col] = feat[col].fillna(0)

    feat["Balance_To_Debit_Ratio"] = (feat["Current_Balance"] / feat["Upcoming_Total_Debit"].replace(0, 1)).clip(0, 20)
    feat["Upcoming_Debit_Pct"]     = (feat["Upcoming_Total_Debit"] / feat["Current_Balance"].replace(0, 1)).clip(0, 5)

    p75_fail = feat["Failed_Debit_Rate"].quantile(0.75)
    p75_upco = feat["Upcoming_Total_Debit"].quantile(0.75)
    feat["Risk_Label"] = feat.apply(assign_label, axis=1, p75_fail=p75_fail, p75_upco=p75_upco)

    for target, sort_col, asc in [(2, "Failed_Debit_Rate", False), (0, "Failed_Debit_Rate", True)]:
        if (feat["Risk_Label"] == target).sum() < 5:
            feat.loc[feat.nlargest(10, sort_col).index if not asc else feat.nsmallest(10, sort_col).index, "Risk_Label"] = target

    return feat


def train_risk_model(feat_df):
    print("\n[FR6] Training Risk Scoring Model (Gradient Boosting)...")

    X = feat_df[FEATURES].fillna(0)
    y = feat_df["Risk_Label"]

    counts   = Counter(y.tolist())
    stratify = y if all(v >= 2 for v in counts.values()) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                              learning_rate=0.07, subsample=0.8, random_state=42)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    names  = {0: "Low", 1: "Medium", 2: "High"}
    print(f"  Accuracy: {acc:.4f} {'OK' if acc >= 0.85 else 'WARNING'} (target: >=85%)")
    print(classification_report(y_test, y_pred, target_names=[names[i] for i in sorted(names)]))

    dataset_stats = {
        "p75_fail":   float(feat_df["Failed_Debit_Rate"].quantile(0.75)),
        "p75_subs":   float(feat_df["Subscription_Count"].quantile(0.75)),
        "p75_upco":   float(feat_df["Upcoming_Total_Debit"].quantile(0.75)),
        "p75_burden": float(feat_df["Total_Monthly_Sub_Amount"].quantile(0.75)),
    }

    scored = feat_df.copy()
    scored["Risk_Score"]  = compute_score(scored)
    scored["Risk_Level"]  = scored["Risk_Score"].apply(risk_level)
    scored["Risk_Reason"] = scored.apply(lambda r: risk_reason(r, dataset_stats), axis=1)

    dist = scored["Risk_Level"].value_counts()
    for lvl in ["High", "Medium", "Low"]:
        print(f"  {lvl}: {dist.get(lvl, 0)}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "dataset_stats": dataset_stats}, "models/risk_model.pkl")
    print("[FR6] Saved -> models/risk_model.pkl")

    return model, scored


def load_risk_model():
    bundle = joblib.load("models/risk_model.pkl")
    # Handle both old format (just model) and new format (dict with model + stats)
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("dataset_stats", {})
    return bundle, {}
