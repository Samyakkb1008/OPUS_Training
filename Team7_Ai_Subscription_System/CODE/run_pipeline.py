"""
run_pipeline.py
AI Subscription & Auto-Debit Intelligence System — Team 7
Full pipeline: FR1 (data) -> FR2 (clean) -> FR3 (NLP) -> FR4 (patterns)
            -> FR5 (predict) -> FR6 (risk) -> FR7 (alerts) -> FR8 (insights) -> FR9 (dashboard PNG)

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-gen          # Skip dataset generation (use existing data)
    python run_pipeline.py --top-n 50          # Generate alerts for top 50 customers
"""

import os, sys, json, time, argparse
import pandas as pd

# Resolve paths relative to this file
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATA    = os.path.join(ROOT, "data")
MODELS  = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")

for d in [DATA, MODELS, REPORTS]:
    os.makedirs(d, exist_ok=True)

# Change CWD so relative paths inside modules resolve correctly
os.chdir(ROOT)


def save(df, name):
    path = os.path.join(DATA, name)
    df.to_csv(path, index=False)
    print(f"  Saved -> {path}  ({len(df):,} rows)")


def main(skip_gen=False, top_n=25):
    start = time.time()
    print("=" * 65)
    print("  AI Subscription & Auto-Debit Intelligence System")
    print("  Team 7 -- Mansi & Samyak")
    print("=" * 65)

    # ── FR1: Generate / load dataset ─────────────────────────────────────────
    raw_path = os.path.join(DATA, "transactions_raw.csv")
    if skip_gen and os.path.exists(raw_path):
        print("\n[FR1] Loading existing dataset...")
        df_raw = pd.read_csv(raw_path)
        print(f"  Loaded {len(df_raw):,} transactions")
    else:
        print("\n[FR1] Generating synthetic dataset...")
        from modules.fr1_dataset_generator import generate_dataset
        df_raw = generate_dataset()
        save(df_raw, "transactions_raw.csv")

    # ── FR2: Data cleaning ────────────────────────────────────────────────────
    print("\n[FR2] Cleaning data...")
    from modules.fr2_data_cleaning import clean_data
    df_clean = clean_data(df_raw)
    save(df_clean, "transactions_cleaned.csv")

    # ── FR3: NLP subscription detection ──────────────────────────────────────
    print("\n[FR3] NLP subscription detection...")
    from modules.fr3_nlp_detector import train_nlp_model, predict_subscriptions
    nlp_model, nlp_metrics = train_nlp_model(df_clean)
    df_nlp = predict_subscriptions(nlp_model, df_clean)
    # Sync SubscriptionFlag with NLP prediction
    df_nlp["SubscriptionFlag"] = df_nlp["NLP_Sub_Pred"]
    save(df_nlp, "transactions_nlp.csv")

    # ── FR4: Recurring pattern detection ─────────────────────────────────────
    print("\n[FR4] Detecting recurring patterns...")
    from modules.fr4_pattern_detector import detect_recurring_patterns
    df_patterns, summary_df, insights_df = detect_recurring_patterns(df_nlp)
    save(df_patterns, "transactions_patterns.csv")
    save(summary_df,  "recurring_summary.csv")

    # ── FR5: Debit prediction ─────────────────────────────────────────────────
    print("\n[FR5] Predicting next debits...")
    from modules.fr5_prediction_engine import predict_next_debits
    pred_df = predict_next_debits(df_patterns)
    save(pred_df, "predictions.csv")

    # ── FR6: Risk scoring ─────────────────────────────────────────────────────
    print("\n[FR6] Risk scoring...")
    from modules.fr6_risk_scoring import build_risk_features, score_with_model
    feat_df   = build_risk_features(df_patterns, pred_df)
    risk_df = score_with_model(feat_df)

    # Keep only needed columns for dashboard
    risk_out_cols = [
        "CustomerID", "Current_Balance", "Failed_Debits", "Total_Debits",
        "Avg_Debit_Amount", "Total_Monthly_Sub_Amount", "Subscription_Count",
        "Failed_Debit_Rate", "Upcoming_Total_Debit", "Balance_To_Debit_Ratio",
        "Upcoming_Debit_Pct", "Risk_Label", "Risk_Score", "Risk_Level", "Risk_Reason",
    ]
    risk_save = risk_df[[c for c in risk_out_cols if c in risk_df.columns]]
    save(risk_save, "risk_scored.csv")

    # ── FR8: Customer insights ─────────────────────────────────────────────────
    print("\n[FR8] Generating customer insights...")
    from modules.fr8_insights import generate_insights
    insights_out = generate_insights(df_patterns, summary_df, risk_df, pred_df)
    save(insights_out, "customer_insights.csv")

    # ── FR7: GenAI alerts ─────────────────────────────────────────────────────
    print(f"\n[FR7] Generating alerts (top {top_n} high/medium risk)...")
    from modules.fr7_genai_alerts import generate_alerts
    alerts = generate_alerts(risk_save, pred_df, insights_out, top_n=top_n)

    alerts_path = os.path.join(REPORTS, "alerts.txt")
    with open(alerts_path, "w", encoding="utf-8") as f:
        for a in alerts:
            f.write(a["alert_text"] + "\n")
    print(f"  Alerts written -> {alerts_path}  ({len(alerts)} alerts)")

    # ── FR9: Static dashboard PNG ──────────────────────────────────────────────
    print("\n[FR9] Generating static dashboard PNG...")
    try:
        from streamlit_app.dashboard import generate_static_dashboard
        png_path = os.path.join(REPORTS, "dashboard.png")
        generate_static_dashboard(
            df=df_patterns, pred_df=pred_df, risk_df=risk_save,
            summary_df=summary_df, insights_df=insights_out,
            output_path=png_path, nlp_metrics=nlp_metrics,
        )
    except Exception as e:
        print(f"  [FR9] Dashboard PNG skipped: {e}")

    # ── Pipeline summary ───────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)
    summary = {
        "project":   "AI Subscription & Auto-Debit Intelligence System",
        "team":      "Team 7 -- Mansi & Samyak",
        "tech_stack": {
            "NLP":       "spaCy + TF-IDF + Logistic Regression",
            "TimeSeries":"ARIMA(1,0,0) + Linear Regression blend",
            "ML":        "Gradient Boosting Classifier",
            "GenAI":     "Microsoft Phi-2 (rule-based fallback)",
            "UI":        "Streamlit + Plotly",
            "DataGen":   "Faker",
        },
        "results": {
            "total_transactions":  int(len(df_patterns)),
            "subscriptions_found": int(df_patterns["SubscriptionFlag"].sum()) if "SubscriptionFlag" in df_patterns.columns else 0,
            "recurring_confirmed": int(df_patterns["Is_Recurring"].sum()) if "Is_Recurring" in df_patterns.columns else 0,
            "predictions_made":    int(len(pred_df)),
            "high_risk":           int((risk_save["Risk_Level"] == "High").sum()),
            "medium_risk":         int((risk_save["Risk_Level"] == "Medium").sum()),
            "low_risk":            int((risk_save["Risk_Level"] == "Low").sum()),
            "alerts":              int(len(alerts)),
        },
        "metrics": {
            "nlp_accuracy":   round(nlp_metrics.get("accuracy", 0), 4),
        },
        "brd_compliance": {
            "FR1": "Synthetic dataset generated (Faker)",
            "FR2": "Data cleaning complete",
            "FR3": f"NLP accuracy {nlp_metrics.get('accuracy', 0)*100:.2f}% (target >90%)",
            "FR4": "Recurring patterns detected (min 3 occurrences)",
            "FR5": "ARIMA(1,0,0) + Linear Regression prediction",
            "FR6": "Gradient Boosting risk scoring (target >85%)",
            "FR7": f"Structured alerts generated ({len(alerts)} alerts)",
            "FR8": "Customer insights and monthly spend summary",
            "FR9": "Streamlit + Plotly interactive dashboard",
        },
        "runtime_seconds": elapsed,
    }

    summary_path = os.path.join(REPORTS, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Pipeline complete in {elapsed}s")
    print(f"  Transactions : {summary['results']['total_transactions']:,}")
    print(f"  Subscriptions: {summary['results']['subscriptions_found']:,}")
    print(f"  Predictions  : {summary['results']['predictions_made']:,}")
    print(f"  High Risk    : {summary['results']['high_risk']:,}")
    print(f"  Alerts       : {summary['results']['alerts']:,}")
    print(f"  NLP Accuracy : {summary['metrics']['nlp_accuracy']*100:.2f}%")
    print(f"{'='*65}")
    print(f"\n  Run dashboard: streamlit run streamlit_app/dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gen", action="store_true", help="Skip dataset generation")
    parser.add_argument("--top-n",    type=int, default=25, help="Number of alerts to generate")
    args = parser.parse_args()
    main(skip_gen=args.skip_gen, top_n=args.top_n)
