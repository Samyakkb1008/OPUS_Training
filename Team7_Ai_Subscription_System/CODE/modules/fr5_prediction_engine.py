"""
FR5 - Next Debit Prediction
Predicts next debit date (Linear Regression + median gap blend)
and amount (ARIMA if available, else EWM).
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_OK = True
except ImportError:
    ARIMA_OK = False
    print("[FR5] statsmodels not found — using EWM fallback.")


def predict_next_date(dates, freq):
    if len(dates) < 2:
        return (dates[-1] + timedelta(days=30 if freq == "Monthly" else 7)).date()

    gaps    = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    med_gap = int(np.median(gaps))

    ordinals = np.array([d.toordinal() for d in dates]).reshape(-1, 1)
    indices  = np.arange(len(dates)).reshape(-1, 1)
    lr       = LinearRegression().fit(indices, ordinals)
    next_ord = int(lr.predict([[len(dates)]])[0][0])
    next_ord = max(next_ord, dates[-1].toordinal() + 1)
    lr_date  = pd.Timestamp.fromordinal(next_ord)

    blend_days = int(0.7 * (lr_date - dates[-1]).days + 0.3 * med_gap)
    return (dates[-1] + timedelta(days=max(blend_days, 1))).date()


def predict_next_amount(amounts):
    # ARIMA(1,0,0) if enough data, else EWM
    if ARIMA_OK and len(amounts) >= 4:
        try:
            result = ARIMA(amounts, order=(1, 0, 0)).fit()
            return round(max(float(result.forecast(steps=1)[0]), 0), 2)
        except:
            pass
    # EWM fallback
    weights = np.exp(np.linspace(0, 1, len(amounts)))
    weights /= weights.sum()
    return round(float(np.dot(weights, amounts)), 2)


def predict_next_debits(df):
    print("\n[FR5] Predicting next debits...")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    recurring = df[(df["Is_Recurring"] == 1) &
                   (df["TransactionType"] == "DEBIT") &
                   (df["Status"] == "SUCCESS")]

    predictions = []
    for (cust_id, desc), group in recurring.groupby(["CustomerID", "Description_Clean"]):
        group   = group.sort_values("Date")
        dates   = group["Date"].tolist()
        amounts = group["Amount"].tolist()
        freq    = group["Inferred_Freq"].iloc[0]

        if len(dates) < 2:
            continue

        next_date   = predict_next_date(dates, freq)
        next_amount = predict_next_amount(amounts)
        method      = "ARIMA(1,0,0)" if (ARIMA_OK and len(amounts) >= 4) else "EWM"

        predictions.append({
            "CustomerID":         cust_id,
            "Subscription":       desc,
            "Merchant":           group["Merchant"].mode().iloc[0] if "Merchant" in group.columns else "—",
            "Frequency":          freq,
            "Occurrences":        len(dates),
            "Last_Debit_Date":    dates[-1].date(),
            "Next_Debit_Date":    next_date,
            "Avg_Historical_Amt": round(np.mean(amounts), 2),
            "Predicted_Amount":   next_amount,
            "Prediction_Method":  method,
        })

    pred_df = pd.DataFrame(predictions)
    print(f"  Predictions generated : {len(pred_df)}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"predictions": pred_df.to_dict(orient="records")}, "models/prediction_model.pkl")
    print("[FR5] Saved → models/prediction_model.pkl")

    return pred_df


def load_prediction_model():
    return joblib.load("models/prediction_model.pkl")


##############################################################################################

# """
# FR5 - Next Debit Prediction
# Predicts the next debit date and amount for each confirmed recurring subscription.
# Date: linear regression + median gap blend (handles Feb/month-end drift)
# Amount: ARIMA(1,0,0) if statsmodels available, else Exponential Weighted Mean
# """

# import os
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import timedelta, date
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score

# # Try ARIMA (statsmodels), fall back to EWM if not installed
# try:
#     from statsmodels.tsa.arima.model import ARIMA
#     ARIMA_OK = True
# except ImportError:
#     ARIMA_OK = False
#     print("[Prediction] ⚠  statsmodels not installed — using EWM fallback.")


# def _predict_next_date(dates, freq):
#     """
#     Predict the next transaction date.
#     Uses a blend of linear regression trend (70%) and median gap (30%).
#     This handles February and month-end irregularities.
#     """
#     if len(dates) < 2:
#         return (dates[-1] + timedelta(days=30 if freq == "Monthly" else 7)).date()

#     gaps    = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
#     med_gap = int(np.median(gaps))

#     # Fit a line through the date ordinals
#     ordinals = np.array([d.toordinal() for d in dates]).reshape(-1, 1)
#     indices  = np.arange(len(dates)).reshape(-1, 1)
#     lr       = LinearRegression().fit(indices, ordinals)
#     next_ord = int(np.ravel(lr.predict([[len(dates)]]))[0])
#     next_ord = max(next_ord, dates[-1].toordinal() + 1)
#     lr_date  = pd.Timestamp.fromordinal(next_ord)

#     # Blend LR prediction with median gap
#     blend_days = int(0.7 * (lr_date - dates[-1]).days + 0.3 * med_gap)
#     return (dates[-1] + timedelta(days=max(blend_days, 1))).date()


# def _predict_next_amount_arima(amounts):
#     """Predict next amount using ARIMA(1,0,0). Falls back to EWM if fails."""
#     if not ARIMA_OK or len(amounts) < 4:
#         return _predict_next_amount_ewm(amounts)
#     try:
#         result = ARIMA(amounts, order=(1, 0, 0)).fit()
#         return round(max(float(result.forecast(steps=1)[0]), 0), 2)
#     except Exception:
#         return _predict_next_amount_ewm(amounts)


# def _predict_next_amount_ewm(amounts):
#     """Predict next amount using exponentially weighted mean (recent = more weight)."""
#     if len(amounts) == 1:
#         return round(amounts[0], 2)
#     weights = np.exp(np.linspace(0, 1, len(amounts)))
#     weights /= weights.sum()
#     return round(float(np.dot(weights, amounts)), 2)


# def predict_next_debits(df):
#     """
#     For every confirmed recurring subscription, predict next debit date and amount.
#     Saves prediction metadata to models/prediction_model.pkl.
#     Returns a DataFrame with one row per customer-subscription pair.
#     """
#     print(f"\n{'='*60}")
#     print("  FR5 - NEXT DEBIT PREDICTION")
#     print(f"  Method: {'ARIMA(1,0,0)' if ARIMA_OK else 'EWM Regression'} + Linear Regression (date)")
#     print(f"{'='*60}")

#     df = df.copy()
#     df["Date"] = pd.to_datetime(df["Date"])

#     # Only use successful recurring debit transactions
#     recurring = df[(df["Is_Recurring"] == 1) &
#                    (df["TransactionType"] == "DEBIT") &
#                    (df["Status"] == "SUCCESS")]

#     predictions = []
#     for (cust_id, desc), group in recurring.groupby(["CustomerID", "Description_Clean"]):
#         group   = group.sort_values("Date")
#         dates   = group["Date"].tolist()
#         amounts = group["Amount"].tolist()
#         freq    = group["Inferred_Freq"].iloc[0]

#         if len(dates) < 2:
#             continue

#         next_date   = _predict_next_date(dates, freq)
#         next_amount = (_predict_next_amount_arima(amounts) if ARIMA_OK and len(amounts) >= 4
#                        else _predict_next_amount_ewm(amounts))
#         method      = "ARIMA(1,0,0)" if ARIMA_OK and len(amounts) >= 4 else "EWM Regression"

#         # Date sequence string for display (BRD FR5 example format)
#         seq = " → ".join(d.strftime("%b %d") for d in dates[-3:])
#         seq += f"  →  {pd.Timestamp(next_date).strftime('%b %d')} (pred)"

#         predictions.append({
#             "CustomerID":         cust_id,
#             "Subscription":       desc,
#             "Merchant":           group["Merchant"].mode().iloc[0] if "Merchant" in group.columns else "—",
#             "Frequency":          freq,
#             "Occurrences":        len(dates),
#             "Last_Debit_Date":    dates[-1].date(),
#             "Next_Debit_Date":    next_date,
#             "Avg_Historical_Amt": round(np.mean(amounts), 2),
#             "Predicted_Amount":   next_amount,
#             "Prediction_Method":  method,
#             "Date_Sequence":      seq,
#         })

#     pred_df = pd.DataFrame(predictions)

#     print(f"\n  Predictions generated : {len(pred_df):,}")
#     if not pred_df.empty:
#         print(f"\n  Sample predictions:")
#         print(pred_df[["CustomerID","Subscription","Frequency",
#                         "Last_Debit_Date","Next_Debit_Date",
#                         "Predicted_Amount","Prediction_Method"]].head(6).to_string(index=False))

#         print(f"\n  Date sequences (BRD FR5 format — last 3 + prediction):")
#         for _, row in pred_df.head(4).iterrows():
#             print(f"    [{row['Subscription'][:35]:<35}]  {row['Date_Sequence']}")

#         # Leave-one-out accuracy evaluation
#         eval_rows = []
#         for (_, __), group in recurring.groupby(["CustomerID", "Description_Clean"]):
#             amts = group.sort_values("Date")["Amount"].tolist()
#             if len(amts) >= 4:
#                 pred_a = (_predict_next_amount_arima(amts[:-1]) if ARIMA_OK
#                           else _predict_next_amount_ewm(amts[:-1]))
#                 eval_rows.append({"actual": amts[-1], "predicted": pred_a})

#         if eval_rows:
#             ev  = pd.DataFrame(eval_rows)
#             mae = mean_absolute_error(ev["actual"], ev["predicted"])
#             r2  = r2_score(ev["actual"], ev["predicted"])
#             print(f"\n  Amount prediction accuracy (leave-one-out):")
#             print(f"    MAE : ₹{mae:.2f}")
#             print(f"    R²  : {r2:.4f}")

#     # Save to pkl
#     models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
#     os.makedirs(models_dir, exist_ok=True)
#     pkl_path = os.path.join(models_dir, "prediction_model.pkl")
#     joblib.dump({"method": f"{'ARIMA(1,0,0)' if ARIMA_OK else 'EWM'} + LR blend",
#                  "predictions": pred_df.to_dict(orient="records"),
#                  "total_predicted": len(pred_df)}, pkl_path)
#     print(f"\n  ✅ Prediction model saved → {pkl_path}")
#     print(f"{'='*60}\n")
#     return pred_df


# def load_prediction_model(models_dir=None):
#     """Load saved prediction metadata from pkl."""
#     if models_dir is None:
#         models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
#     path = os.path.join(models_dir, "prediction_model.pkl")
#     if not os.path.exists(path):
#         raise FileNotFoundError("prediction_model.pkl not found. Run predict_next_debits() first.")
#     return joblib.load(path)


# if __name__ == "__main__":
#     df   = pd.read_csv("../data/transactions_patterns.csv")
#     pred = predict_next_debits(df)
#     print(pred.head())
