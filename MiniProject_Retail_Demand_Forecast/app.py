# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # ----------------------------------
# # Page Config
# # ----------------------------------
# st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

# st.title("📦 Retail Inventory Demand Forecasting")
# st.markdown("Forecast product demand to optimize inventory management")

# # ----------------------------------
# # Load Dataset (hardened)
# # ----------------------------------
# @st.cache_data
# def load_data(path: str = "G6_retail_sales_dataset.csv") -> pd.DataFrame:
#     df = pd.read_csv(path, low_memory=False)
#     # Basic types
#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
#     # Clean text fields used in filters
#     if "Product Category" in df.columns:
#         df["Product Category"] = df["Product Category"].astype(str).str.strip()
#     # Drop invalid core fields
#     df = df.dropna(subset=["Date", "Quantity"])
#     df = df.sort_values("Date").reset_index(drop=True)
#     return df

# df = load_data()

# # ----------------------------------
# # Show Dataset
# # ----------------------------------
# st.subheader("📊 Dataset Preview")
# st.dataframe(df.head(20), use_container_width=True)
# st.write("**Dataset Shape:**", df.shape)

# # ----------------------------------
# # Sidebar Filters
# # ----------------------------------
# st.sidebar.header("🔎 Filters")
# category = st.sidebar.selectbox(
#     "Select Product Category",
#     ["All"] + sorted(df["Product Category"].dropna().unique()) if "Product Category" in df.columns else ["All"]
# )

# filtered_df = df if category == "All" else df[df["Product Category"] == category]

# if filtered_df.empty:
#     st.warning("No data after applying filters. Adjust and try again.")
#     st.stop()

# # ----------------------------------
# # Aggregate Demand (Daily Time Series)
# # ----------------------------------
# daily_demand = (
#     filtered_df.groupby("Date", as_index=False)["Quantity"]
#     .sum()
#     .sort_values("Date")
#     .reset_index(drop=True)
# )

# st.subheader("📈 Daily Demand Data")
# st.dataframe(daily_demand.head(), use_container_width=True)

# # ----------------------------------
# # Feature Engineering (trend + seasonality + history)
# # ----------------------------------
# daily = daily_demand.copy()
# daily["Day"] = np.arange(len(daily))                         # trend
# daily["dow"] = daily["Date"].dt.dayofweek                    # seasonality (0=Mon..6=Sun)

# # Rolling mean (shifted by 1 to avoid leakage)
# daily["qty_ma7"] = daily["Quantity"].rolling(7, min_periods=1).mean().shift(1)
# daily["qty_ma7"] = daily["qty_ma7"].fillna(daily["Quantity"].expanding().mean())

# # Lag-1 feature
# daily["qty_lag1"] = daily["Quantity"].shift(1)
# daily["qty_lag1"] = daily["qty_lag1"].fillna(daily["Quantity"].iloc[:3].mean())

# # One-hot for DOW (drop_first to avoid multicollinearity)
# dow_dummies = pd.get_dummies(daily["dow"], prefix="dow", drop_first=True)

# # Feature matrix
# X = pd.concat([daily[["Day", "qty_ma7", "qty_lag1"]], dow_dummies], axis=1)
# y = daily["Quantity"].astype(float)

# feature_cols = X.columns.tolist()

# # Train/Test Split (80/20 chronological)
# split = int(len(daily) * 0.8)
# X_train, X_test = X.iloc[:split], X.iloc[split:]
# y_train, y_test = y.iloc[:split], y.iloc[split:]
# dates_train = daily["Date"].iloc[:split]
# dates_test  = daily["Date"].iloc[split:]

# # ----------------------------------
# # Train Model
# # ----------------------------------
# model = LinearRegression()
# model.fit(X_train, y_train)

# # ----------------------------------
# # Predictions on Holdout
# # ----------------------------------
# y_pred = model.predict(X_test)

# # Metrics (add sMAPE for stability with small counts)
# def smape(y_true, y_hat, eps=1e-8):
#     denom = (np.abs(y_true) + np.abs(y_hat) + eps) / 2.0
#     return np.mean(np.abs(y_hat - y_true) / denom) * 100

# mae  = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# smp  = smape(y_test.values, y_pred)

# st.subheader("✅ Model Performance")
# col1, col2, col3 = st.columns(3)
# col1.metric("MAE", f"{mae:.2f}")
# col2.metric("RMSE", f"{rmse:.2f}")
# col3.metric("sMAPE (%)", f"{smp:.2f}")

# # ----------------------------------
# # Visualization: Actual vs Predicted
# # ----------------------------------
# st.subheader("📉 Actual vs Predicted Demand")

# fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(dates_train, y_train, label="Train Data", color="#1f77b4")
# ax.plot(dates_test, y_test, label="Actual Demand", color="green")
# ax.plot(dates_test, y_pred, label="Predicted Demand", color="red")
# ax.set_xlabel("Date")
# ax.set_ylabel("Quantity")
# ax.legend()
# ax.grid(alpha=0.2)
# st.pyplot(fig, clear_figure=True)

# # ----------------------------------
# # Future Forecast (Iterative so lags/rolling update correctly)
# # ----------------------------------
# st.subheader("🔮 Future Demand Forecast")

# forecast_days = st.slider("Forecast Days", 7, 60, 30)

# # Start with history arrays we will extend with predictions
# hist_qty = daily["Quantity"].tolist()
# hist_ma7 = daily["qty_ma7"].tolist()  # not strictly needed but useful for intuition

# last_date = daily["Date"].max()
# future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")

# preds = []
# for i, d in enumerate(future_dates):
#     # Compute features for this future day from most recent history (actuals + prior preds)
#     dow = d.dayofweek
#     day_index = int(daily["Day"].iloc[-1]) + (i + 1)

#     # lags/rolling from history that includes our previous predictions
#     lag1 = float(hist_qty[-1]) if len(hist_qty) >= 1 else float(np.mean(hist_qty))
#     window = hist_qty[-7:] if len(hist_qty) >= 7 else hist_qty
#     ma7 = float(np.mean(window)) if len(window) > 0 else 0.0

#     # Build feature row (ensure same columns/order as training)
#     row_base = {"Day": day_index, "qty_ma7": ma7, "qty_lag1": lag1}
#     dow_cols = [c for c in feature_cols if c.startswith("dow_")]
#     dow_row = {c: 0 for c in dow_cols}
#     active = f"dow_{dow}"  # if 'dow_0' was dropped during get_dummies, it may not exist
#     if active in dow_row:
#         dow_row[active] = 1
#     feat = {**row_base, **dow_row}

#     # Ensure all training columns exist
#     for c in feature_cols:
#         if c not in feat:
#             feat[c] = 0

#     future_X = pd.DataFrame([feat])[feature_cols]
#     y_hat = float(model.predict(future_X)[0])
#     y_hat = max(y_hat, 0.0)  # clamp negatives

#     preds.append(y_hat)
#     # Update history with our prediction for next step
#     hist_qty.append(y_hat)

# # Build forecast DataFrame
# forecast_df = pd.DataFrame({
#     "Date": future_dates,
#     "Predicted Demand": np.maximum(np.round(preds).astype(int), 0)
# })

# st.dataframe(forecast_df, use_container_width=True)

# # Plot Forecast
# fig2, ax2 = plt.subplots(figsize=(12, 5))
# ax2.plot(daily["Date"], daily["Quantity"], label="Historical Demand", color="#1f77b4")
# ax2.plot(forecast_df["Date"], forecast_df["Predicted Demand"], label="Forecast", linestyle="--", color="orange")
# ax2.set_xlabel("Date")
# ax2.set_ylabel("Quantity")
# ax2.legend()
# ax2.grid(alpha=0.2)
# st.pyplot(fig2, clear_figure=True)

# # ----------------------------------
# # Business Insights
# # ----------------------------------
# st.subheader("📌 Inventory Insights")
# st.write("""
# - Adding **weekday seasonality** and **short-term history** (lag/rolling) improves accuracy vs a straight-line trend.
# - Forecasts are generated **iteratively**, so each next day uses prior predictions for lag features—critical for daily time series.
# - If you maintain planned price/promotions, we can add those as features to reduce under-prediction on spikes.
# """)

# _______________________________________________________________________________________________________________________________________________________________________________________


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("📦 Retail Inventory Demand Forecasting")
st.markdown("Forecast product demand to optimize inventory management")

# ----------------------------------
# Load Dataset (hardened)
# ----------------------------------
@st.cache_data
def load_data(path: str = "G6_retail_sales_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    if "Product Category" in df.columns:
        df["Product Category"] = df["Product Category"].astype(str).str.strip()
    df = df.dropna(subset=["Date", "Quantity"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data()

# ----------------------------------
# Sidebar Controls
# ----------------------------------
st.sidebar.header("🔎 Filters")
all_categories = sorted(df["Product Category"].dropna().unique().tolist())
category = st.sidebar.selectbox("Select Product Category", ["All"] + all_categories)

st.sidebar.header("📅 Forecast")
forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

# NEW: View mode toggle
view_mode = st.sidebar.radio(
    "View mode",
    ["One chart – all categories", "Three charts – one per category"],
    index=0
)

# Filter
filtered_df = df.copy() if category == "All" else df[df["Product Category"] == category]
if filtered_df.empty:
    st.warning("No data after applying filters. Adjust and try again.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)
st.write("**Dataset Shape:**", filtered_df.shape)

# ----------------------------------
# Helper functions
# ----------------------------------
def build_daily_timeseries(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily quantity and engineer features for modeling."""
    daily = (
        sub_df.groupby("Date", as_index=False)["Quantity"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    # Features
    daily["Day"] = np.arange(len(daily))
    daily["dow"] = daily["Date"].dt.dayofweek  # 0..6 (0=Mon..6=Sun)
    daily["qty_ma7"] = daily["Quantity"].rolling(7, min_periods=1).mean().shift(1)
    daily["qty_ma7"] = daily["qty_ma7"].fillna(daily["Quantity"].expanding().mean())
    daily["qty_lag1"] = daily["Quantity"].shift(1)
    daily["qty_lag1"] = daily["qty_lag1"].fillna(daily["Quantity"].iloc[:3].mean())
    return daily

def make_feature_matrix(daily: pd.DataFrame):
    """Create X, y, feature_cols including one-hot DOW (drop_first)."""
    dow_dummies = pd.get_dummies(daily["dow"], prefix="dow", drop_first=True)
    X = pd.concat([daily[["Day", "qty_ma7", "qty_lag1"]], dow_dummies], axis=1)
    y = daily["Quantity"].astype(float)
    feature_cols = X.columns.tolist()
    return X, y, feature_cols

def chronological_split(X, y, dates, ratio=0.8):
    s = int(len(X) * ratio)
    return X.iloc[:s], X.iloc[s:], y.iloc[:s], y.iloc[s:], dates.iloc[:s], dates.iloc[s:]

def smape(y_true, y_hat, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_hat) + eps) / 2.0
    return float(np.mean(np.abs(y_hat - y_true) / denom) * 100)

def train_and_forecast_category(sub_df: pd.DataFrame, horizon: int):
    """Train the model for a subset (one category), return metrics & forecast_df."""
    daily = build_daily_timeseries(sub_df)
    if len(daily) < 14:
        return None, None, None, None, None  # not enough history

    X, y, feature_cols = make_feature_matrix(daily)
    dates = daily["Date"]

    # Split
    X_train, X_test, y_train, y_test, d_train, d_test = chronological_split(X, y, dates, 0.8)

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on holdout
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    smp = smape(y_test.values, y_pred)

    # Iterative future forecast
    last_date = daily["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    hist_qty = daily["Quantity"].tolist()

    preds = []
    for i, d in enumerate(future_dates):
        dow = d.dayofweek
        day_index = int(daily["Day"].iloc[-1]) + (i + 1)

        # Lags/rolling from latest history (actuals + prior preds)
        lag1 = float(hist_qty[-1]) if len(hist_qty) >= 1 else float(np.mean(hist_qty))
        window = hist_qty[-7:] if len(hist_qty) >= 7 else hist_qty
        ma7 = float(np.mean(window)) if len(window) > 0 else 0.0

        # Build feature row in the same order as training
        row_base = {"Day": day_index, "qty_ma7": ma7, "qty_lag1": lag1}
        dow_cols = [c for c in feature_cols if c.startswith("dow_")]
        dow_row = {c: 0 for c in dow_cols}
        active = f"dow_{dow}"
        if active in dow_row:
            dow_row[active] = 1
        feat = {**row_base, **dow_row}

        # Ensure all training columns exist
        for c in feature_cols:
            if c not in feat:
                feat[c] = 0

        future_X = pd.DataFrame([feat])[feature_cols]
        y_hat = float(model.predict(future_X)[0])
        y_hat = max(y_hat, 0.0)  # clamp negatives
        preds.append(y_hat)
        hist_qty.append(y_hat)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted": np.array(preds),
        "Predicted_int": np.maximum(np.round(preds).astype(int), 0)
    })

    eval_df = pd.DataFrame({"Date": d_test, "Actual": y_test.values, "Predicted": y_pred})
    return daily, (mae, rmse, smp), eval_df, forecast_df, feature_cols

# ----------------------------------
# Run per category
# ----------------------------------
if category == "All":
    cats = all_categories
else:
    cats = [category]

results = {}
for cat in cats:
    sub = filtered_df[filtered_df["Product Category"] == cat] if category == "All" else filtered_df
    daily_cat, metrics, eval_df, forecast_df, feats = train_and_forecast_category(sub, forecast_days)
    if daily_cat is None:
        st.warning(f"Not enough history for category: {cat}. Skipping.")
        continue
    results[cat] = {
        "daily": daily_cat,
        "metrics": metrics,
        "eval": eval_df,
        "forecast": forecast_df,
        "features": feats
    }

if not results:
    st.error("No categories produced results. Try reducing filters or horizon.")
    st.stop()

# ----------------------------------
# Metrics table
# ----------------------------------
st.subheader("✅ Model Performance by Category")
rows = []
for cat, res in results.items():
    mae, rmse, smp = res["metrics"]
    rows.append({"Category": cat, "MAE": mae, "RMSE": rmse, "sMAPE (%)": smp})
met_df = pd.DataFrame(rows).sort_values("RMSE")
st.dataframe(met_df, use_container_width=True)

# ----------------------------------
# Plots
# ----------------------------------
st.subheader("📉 Actual vs Predicted (Holdout) by Category")
for cat, res in results.items():
    eval_df = res["eval"]
    daily_cat = res["daily"]
    split_idx = int(len(daily_cat) * 0.8)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_cat["Date"].iloc[:split_idx], daily_cat["Quantity"].iloc[:split_idx], label=f"{cat} – Train", color="#1f77b4")
    ax.plot(eval_df["Date"], eval_df["Actual"], label=f"{cat} – Actual", color="green")
    ax.plot(eval_df["Date"], eval_df["Predicted"], label=f"{cat} – Predicted", color="red")
    ax.set_xlabel("Date"); ax.set_ylabel("Quantity"); ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True)

# ----------------------------------
# Forecast visualizations
# ----------------------------------
st.subheader("🔮 Future Forecast")

if view_mode == "One chart – all categories":
    # Overlay forecasts for all categories
    fig, ax = plt.subplots(figsize=(12, 5))
    # Show the total historical (optional): sum of all filtered categories
    hist_all = (
        filtered_df.groupby("Date", as_index=False)["Quantity"].sum().sort_values("Date")
    )
    ax.plot(hist_all["Date"], hist_all["Quantity"], label="Historical (All filtered)", color="#999999")

    for cat, res in results.items():
        ax.plot(res["forecast"]["Date"], res["forecast"]["Predicted_int"], label=f"{cat} – Forecast")

    ax.set_xlabel("Date"); ax.set_ylabel("Quantity")
    ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True)

else:
    # Three separate charts (or N charts if >3 categories)
    for cat, res in results.items():
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        # Historical for this category
        daily_cat = res["daily"]
        ax2.plot(daily_cat["Date"], daily_cat["Quantity"], label=f"{cat} – Historical", color="#1f77b4")
        # Forecast
        ax2.plot(res["forecast"]["Date"], res["forecast"]["Predicted_int"], label=f"{cat} – Forecast", linestyle="--", color="orange")
        ax2.set_xlabel("Date"); ax2.set_ylabel("Quantity")
        ax2.legend(); ax2.grid(alpha=0.2)
        st.pyplot(fig2, clear_figure=True)

# ----------------------------------
# VISUALIZATIONS
# ----------------------------------
st.subheader("📈 Forecast Graphs")

if view_mode == "One chart – all categories":
    fig, ax = plt.subplots(figsize=(14, 6))

    for cat, res in results.items():
        daily = res["daily"]
        fc = res["forecast"]
        ax.plot(daily["Date"], daily["Quantity"], alpha=0.3)
        ax.plot(fc["Date"], fc["Predicted Demand"], "--", label=f"{cat}")

    ax.set_title("Forecast for All Categories")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

else:
    for cat, res in results.items():
        daily = res["daily"]
        fc = res["forecast"]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(daily["Date"], daily["Quantity"], label="Historical", color="#1f77b4")
        ax.plot(fc["Date"], fc["Predicted Demand"], "--", label="Forecast", color="orange")
        
        ax.set_title(f"{cat} – Forecast")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# ----------------------------------
# Notes
# ----------------------------------
st.caption("""
Notes:
- Each category is trained independently with the same features (trend, DOW, lag-1, 7-day MA) and **iterative** forecasting.
- In "All" mode, the overlay chart shows all categories' forecasts together. Switch to "Three charts" for separate panels.
- You can add planned prices/promotions per category later to improve spike days.
""")

# ________________________________________________________________________________________________________________________________________________________________________________________________

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # ----------------------------------
# # Page Config
# # ----------------------------------
# st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

# st.title("📦 Retail Inventory Demand Forecasting")
# st.markdown("Forecast product demand to optimize inventory management")

# # ----------------------------------
# # Load Dataset
# # ----------------------------------
# @st.cache_data
# def load_data(path: str = "G6_retail_sales_dataset.csv") -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
#     df["Product Category"] = df["Product Category"].astype(str).str.strip()
#     df = df.dropna(subset=["Date", "Quantity"])
#     df = df.sort_values("Date").reset_index(drop=True)
#     return df

# df = load_data()

# # ----------------------------------
# # Sidebar Controls
# # ----------------------------------
# st.sidebar.header("🔎 Filters")
# categories = sorted(df["Product Category"].unique().tolist())

# category_option = st.sidebar.selectbox(
#     "Select Category",
#     ["All"] + categories
# )

# forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

# view_mode = st.sidebar.radio(
#     "View Mode",
#     ["One chart – all categories", "Separate charts – each category"]
# )

# # Apply filter
# filtered_df = df if category_option == "All" else df[df["Product Category"] == category_option]

# # ----------------------------------
# # Helper Functions
# # ----------------------------------
# def build_daily(sub_df):
#     daily = (
#         sub_df.groupby("Date", as_index=False)["Quantity"]
#         .sum()
#         .sort_values("Date")
#         .reset_index(drop=True)
#     )
#     daily["Day"] = np.arange(len(daily))
#     daily["dow"] = daily["Date"].dt.dayofweek

#     daily["qty_ma7"] = daily["Quantity"].rolling(7, min_periods=1).mean().shift(1)
#     daily["qty_ma7"] = daily["qty_ma7"].fillna(daily["Quantity"].expanding().mean())

#     daily["qty_lag1"] = daily["Quantity"].shift(1)
#     daily["qty_lag1"] = daily["qty_lag1"].fillna(daily["Quantity"].iloc[:3].mean())
#     return daily

# def create_features(daily):
#     dow_dummies = pd.get_dummies(daily["dow"], prefix="dow", drop_first=True)
#     X = pd.concat([daily[["Day", "qty_ma7", "qty_lag1"]], dow_dummies], axis=1)
#     y = daily["Quantity"]
#     return X, y, X.columns.tolist()

# def split_ts(X, y, daily, ratio=0.8):
#     split = int(len(X) * ratio)
#     return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], \
#            daily["Date"].iloc[:split], daily["Date"].iloc[split:]

# def smape(y_true, y_pred):
#     return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)

# def forecast_future(daily, model, feature_cols, horizon):
#     last_date = daily["Date"].max()
#     future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

#     hist_qty = daily["Quantity"].tolist()
#     preds = []

#     for i, d in enumerate(future_dates):
#         dow = d.dayofweek
#         next_day = daily["Day"].iloc[-1] + i + 1

#         lag1 = hist_qty[-1]
#         window = hist_qty[-7:] if len(hist_qty) >= 7 else hist_qty
#         ma7 = float(np.mean(window))

#         row = {"Day": next_day, "qty_ma7": ma7, "qty_lag1": lag1}

#         dow_cols = [c for c in feature_cols if c.startswith("dow_")]
#         dow_map = {c: 0 for c in dow_cols}
#         active = f"dow_{dow}"
#         if active in dow_map:
#             dow_map[active] = 1

#         feat = {**row, **dow_map}
#         for col in feature_cols:
#             if col not in feat:
#                 feat[col] = 0

#         future_X = pd.DataFrame([feat])[feature_cols]
#         pred = max(0, model.predict(future_X)[0])
#         preds.append(pred)
#         hist_qty.append(pred)

#     df_fc = pd.DataFrame({
#         "Date": future_dates,
#         "Predicted Demand": np.round(preds).astype(int)
#     })
#     return df_fc

# # ----------------------------------
# # PER CATEGORY MODEL EXECUTION
# # ----------------------------------
# if category_option == "All":
#     categories_to_run = categories
# else:
#     categories_to_run = [category_option]

# results = {}

# for cat in categories_to_run:
#     sub_df = df[df["Product Category"] == cat]
#     daily = build_daily(sub_df)
#     if len(daily) < 14:
#         continue

#     X, y, feature_cols = create_features(daily)
#     X_train, X_test, y_train, y_test, d_train, d_test = split_ts(X, y, daily)

#     model = LinearRegression().fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     smp = smape(y_test.values, y_pred)

#     forecast_df = forecast_future(daily, model, feature_cols, forecast_days)

#     results[cat] = {
#         "daily": daily,
#         "forecast": forecast_df,
#         "metrics": (mae, rmse, smp),
#         "d_train": d_train,
#         "d_test": d_test,
#         "y_train": y_train,
#         "y_test": y_test,
#         "y_pred": y_pred
#     }

# # ----------------------------------
# # Show metrics
# # ----------------------------------
# st.subheader("📊 Model Performance by Category")
# met_list = []
# for cat, res in results.items():
#     mae, rmse, smp = res["metrics"]
#     met_list.append([cat, mae, rmse, smp])

# df_met = pd.DataFrame(met_list, columns=["Category", "MAE", "RMSE", "sMAPE"])
# st.dataframe(df_met, use_container_width=True)

# # ----------------------------------
# # EXACT PREDICTED VALUES DISPLAY
# # ----------------------------------
# st.subheader("🔢 Exact Forecasted Values")

# for cat, res in results.items():
#     st.write(f"### {cat} – Forecasted Values")
#     st.dataframe(res["forecast"], use_container_width=True)

#     # Tomorrow's forecast
#     tomorrow = res["forecast"]["Predicted Demand"].iloc[0]
#     st.metric(f"Tomorrow's Demand – {cat}", int(tomorrow))

# # ----------------------------------
# # VISUALIZATIONS
# # ----------------------------------
# st.subheader("📈 Forecast Graphs")

# if view_mode == "One chart – all categories":
#     fig, ax = plt.subplots(figsize=(14, 6))

#     for cat, res in results.items():
#         daily = res["daily"]
#         fc = res["forecast"]
#         ax.plot(daily["Date"], daily["Quantity"], alpha=0.3)
#         ax.plot(fc["Date"], fc["Predicted Demand"], "--", label=f"{cat}")

#     ax.set_title("Forecast for All Categories")
#     ax.legend()
#     ax.grid(alpha=0.3)
#     st.pyplot(fig)

# else:
#     for cat, res in results.items():
#         daily = res["daily"]
#         fc = res["forecast"]

#         fig, ax = plt.subplots(figsize=(12, 5))
#         ax.plot(daily["Date"], daily["Quantity"], label="Historical", color="#1f77b4")
#         ax.plot(fc["Date"], fc["Predicted Demand"], "--", label="Forecast", color="orange")
        
#         ax.set_title(f"{cat} – Forecast")
#         ax.legend()
#         ax.grid(alpha=0.3)
#         st.pyplot(fig)