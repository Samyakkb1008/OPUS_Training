# """
# FR9 - Streamlit Dashboard
# AI Subscription & Auto-Debit Intelligence System
# Team 7 - Mansi & Samyak

# Tabs: Active Subscriptions | Upcoming Debits | Risk Alerts | Monthly Spend | GenAI Alerts | Add New Customer
# Run: streamlit run streamlit_app/dashboard.py
# """

# import os, re, json
# from datetime import date, timedelta

# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go

# BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA = os.path.join(BASE, "data")
# RPT  = os.path.join(BASE, "reports")

# # ── FR10 Chatbot ──────────────────────────────────────────────────────────────
# import sys
# sys.path.insert(0, BASE)
# try:
#     from modules.fr10_chatbot import chat, init_chatbot
#     _CHATBOT_READY = init_chatbot()
# except Exception as _e:
#     print(f"[Dashboard] Chatbot import failed: {_e}")
#     _CHATBOT_READY = False
#     def chat(q, data, customer_id=None):
#         return {"answer": "Chatbot module not found. Place fr10_chatbot.py in modules/.",
#                 "intent": "error", "model": "none", "latency_ms": 0}

# USERS       = {"banker01": "pass123", "banker02": "pass456", "admin": "admin123"}
# RISK_COLORS = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}

# SUB_WORDS = {
#     "netflix","spotify","amazon","youtube","hotstar","apple","google","microsoft",
#     "linkedin","dropbox","adobe","zoom","coursera","swiggy","subscription","premium",
#     "membership","monthly","weekly","annual","renewal","electricity","utility",
#     "prime","recurring","gym","plan","zee5","bescom","tatasky","jiocinema",
# }
# FP_WORDS = {"salary","wages","refund","dividend","interest","deposit","bonus","neft","imps","rtgs","credit"}


# # ══════════════════════════════════════════════════════════════════════════════
# # DATA LOADER
# # ══════════════════════════════════════════════════════════════════════════════
# @st.cache_data(show_spinner="Loading data...")
# def load_data():
#     files = {
#         "raw":      "transactions_raw.csv",
#         "patterns": "transactions_patterns.csv",
#         "pred":     "predictions.csv",
#         "risk":     "risk_scored.csv",
#         "summary":  "recurring_summary.csv",
#         "insights": "customer_insights.csv",
#     }
#     data = {}
#     for key, fname in files.items():
#         path = os.path.join(DATA, fname)
#         if os.path.exists(path):
#             df = pd.read_csv(path)
#             for col in ["Date", "Last_Date", "Next_Debit_Date", "Last_Debit_Date"]:
#                 if col in df.columns:
#                     df[col] = pd.to_datetime(df[col], errors="coerce")
#             data[key] = df
#         else:
#             data[key] = pd.DataFrame()
#     return data


# # ══════════════════════════════════════════════════════════════════════════════
# # INLINE PIPELINE HELPERS (Add New Customer)
# # ══════════════════════════════════════════════════════════════════════════════
# def nlp_classify(text):
#     tokens = set(re.sub(r"[^a-z\s]", " ", text.lower()).split())
#     if tokens & FP_WORDS:  return False, 0.02
#     if tokens & SUB_WORDS: return True,  0.97
#     return False, 0.40


# def predict_next(date_strs, amount):
#     dates = sorted(pd.Timestamp(str(d)) for d in date_strs)
#     gaps  = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
#     gap   = int(np.median(gaps))
#     freq  = "Weekly" if gap <= 9 else ("Monthly" if gap <= 35 else "Irregular")
#     return (dates[-1] + timedelta(days=gap)).date(), round(float(amount), 2), freq


# def compute_risk(balance, upcoming, fail_rate=0.0, sub_count=1):
#     p_fail   = min(fail_rate / max(0.07, fail_rate + 0.001), 1.0)
#     p_subs   = min(sub_count / 4.0, 1.0)
#     p_upco   = min(upcoming / max(upcoming * 1.5, 1), 1.0)
#     p_burden = min((upcoming / max(balance, 1)) / 0.05, 1.0)
#     score    = round(float(np.clip(
#         0.35 * p_fail + 0.30 * p_subs + 0.20 * p_upco + 0.15 * p_burden, 0, 1)), 4)
#     level    = "High" if score >= 0.65 else ("Medium" if score >= 0.35 else "Low")
#     reasons  = []
#     if fail_rate > 0.05:
#         reasons.append(f"Failure rate {fail_rate*100:.1f}%")
#     if sub_count >= 3:
#         reasons.append(f"{sub_count} active subscriptions")
#     if upcoming > 0:
#         upco_pct = (upcoming / max(balance, 1)) * 100
#         reasons.append(f"Upcoming debit Rs{upcoming:,.0f} ({upco_pct:.1f}% of balance)")
#     reason = "; ".join(reasons) if reasons else f"{sub_count} subscription(s), Rs{upcoming:,.0f} upcoming"
#     return score, level, reason


# def build_alert(cid, merchant, next_date, next_amt, balance, score, level, reason):
#     days      = (pd.Timestamp(str(next_date)).date() - date.today()).days
#     timing    = f"in {days} day(s)" if days > 0 else "TODAY" if days == 0 else f"overdue {abs(days)}d"
#     shortfall = max(0, next_amt - balance)
#     if level == "High":
#         sug = f"Top up Rs{shortfall:,.0f} immediately. Debit due {timing}."
#     elif level == "Medium":
#         sug = f"Maintain Rs{next_amt*1.1:,.0f} before {next_date}."
#     else:
#         sug = "Account balance sufficient. No action needed."
#     return (
#         f"{'='*56}\n  CUSTOMER: {cid}\n{'='*56}\n"
#         f"\n[FR7] ALERT\n"
#         f"  Upcoming Debit: {merchant} Rs{next_amt:,.2f} | "
#         f"Risk: {level} ({score}) | Suggestion: {sug}\n"
#         f"\n[FR6] RISK ASSESSMENT\n"
#         f"  Score  : {score} | Level: {level}\n"
#         f"  Reason : {reason}\n"
#         f"  Balance: Rs{balance:,.2f} | Upcoming: Rs{next_amt:,.2f}\n"
#         f"\n[FR8] INSIGHT\n"
#         f"  1 active subscription | Monthly spend: Rs{next_amt:,.2f}\n"
#         f"{'='*56}"
#     )


# def validate_inputs(cid, desc, merchant, amt, bal, dates):
#     errs = []
#     if not re.match(r"^[A-Za-z0-9_\-]{3,20}$", cid.strip()):
#         errs.append("CustomerID must be 3-20 alphanumeric characters.")
#     if len(desc.strip()) < 3:
#         errs.append("Description must be at least 3 characters.")
#     if not merchant.strip():
#         errs.append("Merchant Name is required.")
#     if amt <= 0:
#         errs.append("Amount must be > Rs0.")
#     if bal < 0:
#         errs.append("Balance cannot be negative.")
#     if len(dates) < 3:
#         errs.append(f"Need at least 3 past dates (BRD FR4). Provided: {len(dates)}.")
#     return errs


# # ══════════════════════════════════════════════════════════════════════════════
# # LOGIN PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# def login_page():
#     st.markdown("<br><br>", unsafe_allow_html=True)
#     col1, col2, col3 = st.columns([1, 1.4, 1])
#     with col2:
#         st.markdown("### 🏦 AI Subscription Intelligence")
#         st.markdown("**SubIntel — Banker Portal**")
#         st.divider()
#         with st.form("login_form"):
#             uid = st.text_input("Banker ID", placeholder="banker01")
#             pwd = st.text_input("Password", type="password")
#             if st.form_submit_button("Sign In", type="primary", use_container_width=True):
#                 if uid in USERS and USERS[uid] == pwd:
#                     st.session_state.update(logged_in=True, banker_id=uid, page="Home")
#                     st.rerun()
#                 else:
#                     st.error("Invalid Banker ID or Password.")
#         st.caption("Demo credentials: banker01 / pass123")


# # ══════════════════════════════════════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════════════════════════════════════
# def render_sidebar():
#     with st.sidebar:
#         st.markdown("## 🏦 SubIntel")
#         st.markdown(f"👤 **{st.session_state.get('banker_id', '')}**")
#         st.divider()
#         if st.button("🏠 Home",     use_container_width=True):
#             st.session_state["page"] = "Home";  st.rerun()
#         if st.button("ℹ️ About",    use_container_width=True):
#             st.session_state["page"] = "About"; st.rerun()
#         st.divider()
#         if st.button("🚪 Sign Out", use_container_width=True):
#             for k in ["logged_in", "banker_id", "page"]:
#                 st.session_state.pop(k, None)
#             st.rerun()


# # ══════════════════════════════════════════════════════════════════════════════
# # GLOBAL FILTERS
# # Single unified dropdown (Customer or Merchant) + date range tuple input
# # ══════════════════════════════════════════════════════════════════════════════
# def render_filters(data):
#     raw     = data.get("raw",     pd.DataFrame())
#     summary = data.get("summary", pd.DataFrame())
#     risk    = data.get("risk",    pd.DataFrame())

#     # Customer options
#     cust_options = ["All Customers"]
#     if not risk.empty and "CustomerID" in risk.columns:
#         cust_options += sorted(risk["CustomerID"].unique().tolist())

#     # Merchant options
#     merch_options = ["All Merchants"]
#     if not summary.empty and "Merchant" in summary.columns:
#         merch_options += sorted(summary["Merchant"].dropna().unique().tolist())

#     # Actual data date range
#     min_d = max_d = date.today()
#     if not raw.empty and "Date" in raw.columns:
#         vd = raw["Date"].dropna()
#         if not vd.empty:
#             min_d, max_d = vd.min().date(), vd.max().date()

#     c1, c2, c3 = st.columns([2, 2, 2])
#     with c1:
#         st.markdown("👤 **Customer**")
#         sel_cust = st.selectbox(
#             "Customer",
#             cust_options,
#             label_visibility="collapsed",
#             help="Filter by a specific customer"
#         )
#     with c2:
#         st.markdown("🏪 **Merchant**")
#         sel_merch = st.selectbox(
#             "Merchant",
#             merch_options,
#             label_visibility="collapsed",
#             help="Filter by a specific merchant"
#         )
#     with c3:
#         st.markdown("📅 **Date Range**")
#         date_range = st.date_input(
#             "Date Range",
#             value=(min_d, max_d),
#             min_value=min_d,
#             max_value=max_d,
#             label_visibility="collapsed",
#         )

#     if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
#         d_from, d_to = date_range[0], date_range[1]
#     else:
#         d_from, d_to = min_d, max_d

#     return sel_cust, sel_merch, d_from, d_to

# # def render_filters(data):
# #     raw     = data.get("raw",     pd.DataFrame())
# #     summary = data.get("summary", pd.DataFrame())
# #     risk    = data.get("risk",    pd.DataFrame())

# #     # Build one unified list: All Data, Customer: X, Merchant: Y
# #     options = ["All Data"]
# #     if not risk.empty and "CustomerID" in risk.columns:
# #         options += [f"Customer: {c}" for c in sorted(risk["CustomerID"].unique().tolist())]
# #     if not summary.empty and "Merchant" in summary.columns:
# #         options += [f"Merchant: {m}" for m in sorted(summary["Merchant"].dropna().unique().tolist())]

# #     # Actual data date range
# #     min_d = max_d = date.today()
# #     if not raw.empty and "Date" in raw.columns:
# #         vd = raw["Date"].dropna()
# #         if not vd.empty:
# #             min_d, max_d = vd.min().date(), vd.max().date()

# #     c1, c2 = st.columns([2, 2])
# #     with c1:
# #         chosen = st.selectbox(
# #             "🔍 Filter by Customer or Merchant",
# #             options,
# #             help="Choose a specific customer or merchant, or keep 'All Data' to see everything"
# #         )
# #     with c2:
# #         # Tuple date_input avoids the from > to bug — user picks both ends at once
# #         date_range = st.date_input(
# #             "📅 Date Range",
# #             value=(min_d, max_d),
# #             min_value=min_d,
# #             max_value=max_d,
# #         )

# #     # date_input returns a tuple once both ends are selected, else a single date
# #     if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
# #         d_from, d_to = date_range[0], date_range[1]
# #     else:
# #         # User is mid-selection; hold previous values
# #         d_from, d_to = min_d, max_d

# #     # Decode the selection
# #     if chosen == "All Data":
# #         sel_cust, sel_merch = "All Customers", "All Merchants"
# #     elif chosen.startswith("Customer: "):
# #         sel_cust  = chosen[len("Customer: "):]
# #         sel_merch = "All Merchants"
# #     elif chosen.startswith("Merchant: "):
# #         sel_cust  = "All Customers"
# #         sel_merch = chosen[len("Merchant: "):]
# #     else:
# #         sel_cust, sel_merch = "All Customers", "All Merchants"

# #     return sel_cust, sel_merch, d_from, d_to


# # ══════════════════════════════════════════════════════════════════════════════
# # KPI ROW
# # ══════════════════════════════════════════════════════════════════════════════
# def render_kpis(data, sel_cust, d_from, d_to):
#     raw  = data.get("raw",  pd.DataFrame())
#     risk = data.get("risk", pd.DataFrame())
#     pred = data.get("pred", pd.DataFrame())

#     rf = raw.copy()
#     if not rf.empty and "Date" in rf.columns:
#         rf = rf[(rf["Date"].dt.date >= d_from) & (rf["Date"].dt.date <= d_to)]
#     if sel_cust != "All Customers" and "CustomerID" in rf.columns:
#         rf = rf[rf["CustomerID"] == sel_cust]

#     txn  = len(rf)
#     subs = int(rf["SubscriptionFlag"].sum()) if "SubscriptionFlag" in rf.columns else 0
#     high = int((risk["Risk_Level"] == "High").sum())   if not risk.empty and "Risk_Level" in risk.columns else 0
#     med  = int((risk["Risk_Level"] == "Medium").sum()) if not risk.empty and "Risk_Level" in risk.columns else 0
#     upco = float(pred["Predicted_Amount"].sum())       if not pred.empty and "Predicted_Amount" in pred.columns else 0

#     k1, k2, k3, k4, k5 = st.columns(5)
#     k1.metric("Total Transactions",  f"{txn:,}")
#     k2.metric("Subscriptions Found", f"{subs:,}")
#     k3.metric("🔴 High Risk",        f"{high:,}")
#     k4.metric("🟡 Medium Risk",      f"{med:,}")
#     k5.metric("💰 Upcoming Debits",  f"Rs{upco:,.0f}")


# # ══════════════════════════════════════════════════════════════════════════════
# # GAUGE CHART HELPER  (used in Risk Alerts for single-customer/merchant view)
# # ══════════════════════════════════════════════════════════════════════════════
# def render_gauge(score, level, title="Risk Score"):
#     color = RISK_COLORS.get(level, "#aaa")
#     fig   = go.Figure(go.Indicator(
#         mode  = "gauge+number+delta",
#         value = round(score, 4),
#         title = {"text": title, "font": {"size": 16}},
#         delta = {"reference": 0.35,
#                  "increasing": {"color": "#e74c3c"},
#                  "decreasing": {"color": "#2ecc71"}},
#         gauge = {
#             "axis":  {"range": [0, 1], "tickwidth": 1},
#             "bar":   {"color": color},
#             "steps": [
#                 {"range": [0,    0.35], "color": "#0d2118"},
#                 {"range": [0.35, 0.65], "color": "#2d2108"},
#                 {"range": [0.65, 1.0],  "color": "#2d1012"},
#             ],
#             "threshold": {
#                 "line":      {"color": color, "width": 4},
#                 "thickness": 0.75,
#                 "value":     score,
#             },
#         },
#     ))
#     fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
#     return fig


# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 1 — ACTIVE SUBSCRIPTIONS
# # ══════════════════════════════════════════════════════════════════════════════
# def tab_subscriptions(data, sel_cust, sel_merch, d_from, d_to):
#     summary = data.get("summary", pd.DataFrame())
#     if summary.empty:
#         st.info("Run `python run_pipeline.py` to generate subscription data."); return

#     sf = summary.copy()
#     if "Last_Date" in sf.columns:
#         sf = sf[(sf["Last_Date"].dt.date >= d_from) & (sf["Last_Date"].dt.date <= d_to)]
#     if sel_merch != "All Merchants" and "Merchant" in sf.columns:
#         sf = sf[sf["Merchant"] == sel_merch]
#     if sel_cust != "All Customers" and "CustomerID" in sf.columns:
#         sf = sf[sf["CustomerID"] == sel_cust]

#     if sf.empty:
#         st.warning("No subscriptions match current filters."); return

#     # BRD-required metrics only — clean single row
#     k1, k2, k3 = st.columns(3)
#     k1.metric("Recurring Groups",  f"{len(sf):,}")
#     k2.metric("Unique Customers",  f"{sf['CustomerID'].nunique():,}")
#     k3.metric("Avg Monthly Spend", f"Rs{sf['Avg_Amount'].mean():,.0f}" if "Avg_Amount" in sf.columns else "-")

#     st.divider()

#     sort_col = st.selectbox("Sort by", ["Avg_Amount", "Occurrences", "Frequency", "Median_Gap_Days"])
#     sf = sf.sort_values(sort_col, ascending=False)

#     cols = [c for c in ["CustomerID", "Description", "Merchant", "Frequency",
#                          "Occurrences", "Avg_Amount", "Median_Gap_Days", "Failed_Count"] if c in sf.columns]
#     st.dataframe(
#         sf[cols].rename(columns={
#             "Avg_Amount":      "Avg Amount (Rs)",
#             "Median_Gap_Days": "Gap (days)",
#             "Failed_Count":    "Failures",
#         }),
#         use_container_width=True, hide_index=True
#     )
#     st.download_button("⬇️ Export CSV", data=sf[cols].to_csv(index=False),
#                        file_name="active_subscriptions.csv", mime="text/csv")

#     # Chart 1 — Top subscriptions bar chart (BRD required)
#     st.subheader("Top Subscriptions by Occurrence")
#     top = sf.groupby("Description")["Occurrences"].sum() \
#             .sort_values(ascending=False).head(10).reset_index()
#     fig = px.bar(top, x="Occurrences", y="Description", orientation="h",
#                  color="Occurrences", color_continuous_scale="Blues",
#                  labels={"Description": ""})
#     fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
#     st.plotly_chart(fig, use_container_width=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 2 — UPCOMING DEBITS
# # ══════════════════════════════════════════════════════════════════════════════
# def tab_upcoming(data, sel_cust, sel_merch, d_from, d_to):
#     pred = data.get("pred", pd.DataFrame())
#     if pred.empty:
#         st.info("Run the pipeline to generate predictions."); return

#     pf = pred.copy()
#     if "Next_Debit_Date" in pf.columns:
#         next30 = date.today() + timedelta(days=30)
#         pf = pf[(pf["Next_Debit_Date"].dt.date >= d_from) &
#                 (pf["Next_Debit_Date"].dt.date <= next30)]
#     if sel_merch != "All Merchants" and "Merchant" in pf.columns:
#         pf = pf[pf["Merchant"] == sel_merch]
#     if sel_cust != "All Customers" and "CustomerID" in pf.columns:
#         pf = pf[pf["CustomerID"] == sel_cust]

#     if pf.empty:
#         st.warning("No upcoming debits in the next 30 days for current filters."); return

#     sort_col = st.selectbox("Sort by", ["Next_Debit_Date", "Predicted_Amount", "Frequency"])
#     pf = pf.sort_values(sort_col)

#     cols = [c for c in ["CustomerID", "Subscription", "Merchant", "Frequency",
#                          "Next_Debit_Date", "Predicted_Amount", "Prediction_Method"] if c in pf.columns]
#     st.dataframe(
#         pf[cols].rename(columns={"Next_Debit_Date": "Next Debit", "Predicted_Amount": "Amount (Rs)"}),
#         use_container_width=True, hide_index=True
#     )
#     st.download_button("⬇️ Export CSV", data=pf[cols].to_csv(index=False),
#                        file_name="upcoming_debits.csv", mime="text/csv")

#     # Chart 4 — Debit Timeline (BRD required)
#     st.subheader("Debit Timeline")
#     if "Next_Debit_Date" in pf.columns and "Predicted_Amount" in pf.columns:
#         tl  = pf.sort_values("Next_Debit_Date")
#         fig = px.line(tl, x="Next_Debit_Date", y="Predicted_Amount",
#                       color="Merchant" if "Merchant" in tl.columns else None,
#                       markers=True,
#                       labels={"Next_Debit_Date": "Date", "Predicted_Amount": "Amount (Rs)"},
#                       title="Upcoming Debit Timeline (next 30 days)")
#         st.plotly_chart(fig, use_container_width=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 3 — RISK ALERTS
# # - Counts per risk level shown after filter
# # - Pie chart (or gauge for single customer) shown ABOVE the cards
# # - Risk Score Distribution histogram REMOVED
# # ══════════════════════════════════════════════════════════════════════════════
# def tab_risk(data, sel_cust, sel_merch):
#     risk = data.get("risk", pd.DataFrame())
#     pred = data.get("pred", pd.DataFrame())
#     if risk.empty:
#         st.info("Run the pipeline to generate risk scores."); return

#     rf = risk.copy()
#     if sel_cust != "All Customers" and "CustomerID" in rf.columns:
#         rf = rf[rf["CustomerID"] == sel_cust]
#     if sel_merch != "All Merchants" and not pred.empty and "Merchant" in pred.columns:
#         rf = rf[rf["CustomerID"].isin(
#             pred[pred["Merchant"] == sel_merch]["CustomerID"].unique()
#         )]

#     lvl_filter = st.multiselect(
#         "Filter by Risk Level", ["High", "Medium", "Low"], default=["High", "Medium"]
#     )
#     if lvl_filter:
#         rf = rf[rf["Risk_Level"].isin(lvl_filter)]
#     rf = rf.sort_values("Risk_Score", ascending=False)

#     if rf.empty:
#         st.info("No risk alerts for the selected filters."); return

#     # Count per risk level after applying all filters
#     counts = rf["Risk_Level"].value_counts()
#     ck1, ck2, ck3 = st.columns(3)
#     ck1.metric("🔴 High",   f"{counts.get('High',   0):,}")
#     ck2.metric("🟡 Medium", f"{counts.get('Medium', 0):,}")
#     ck3.metric("🟢 Low",    f"{counts.get('Low',    0):,}")

#     st.divider()

#     # Chart: gauge for single customer, pie for all/merchant
#     is_single_customer = (sel_cust != "All Customers")

#     if is_single_customer and not rf.empty:
#         row   = rf.iloc[0]
#         score = float(row.get("Risk_Score", 0))
#         level = str(row.get("Risk_Level", "Low"))
#         cid   = str(row.get("CustomerID", ""))
#         st.subheader(f"Risk Score — {cid}")
#         st.plotly_chart(render_gauge(score, level, title=f"{cid} | {level} Risk"),
#                         use_container_width=True)
#     else:
#         # Pie chart — Chart 3 (BRD required)
#         st.subheader("Risk Distribution")
#         pie = rf["Risk_Level"].value_counts().reset_index()
#         pie.columns = ["Risk_Level", "Count"]
#         fig = px.pie(pie, names="Risk_Level", values="Count",
#                      color="Risk_Level", color_discrete_map=RISK_COLORS)
#         fig.update_traces(textposition="inside", textinfo="percent+label")
#         st.plotly_chart(fig, use_container_width=True)

#     st.divider()

#     # Risk alert cards
#     for _, row in rf.head(20).iterrows():
#         lvl    = str(row.get("Risk_Level", "Low"))
#         score  = float(row.get("Risk_Score", 0))
#         reason = str(row.get("Risk_Reason", "-"))
#         cid    = str(row.get("CustomerID", "-"))
#         bal    = float(row.get("Current_Balance", 0))
#         upco   = float(row.get("Upcoming_Total_Debit", 0))
#         subs   = int(row.get("Subscription_Count", 0))
#         fail_r = float(row.get("Failed_Debit_Rate", 0))
#         icon   = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(lvl, "⚪")

#         with st.container(border=True):
#             a, b = st.columns([4, 1])
#             with a:
#                 st.markdown(f"**{icon} {cid}** — `{lvl} Risk`")
#                 st.caption(f"📋 Reason: {reason}")
#                 c1, c2, c3, c4 = st.columns(4)
#                 c1.markdown(f"**Balance:** Rs{bal:,.0f}")
#                 c2.markdown(f"**Upcoming:** Rs{upco:,.0f}")
#                 c3.markdown(f"**Subscriptions:** {subs}")
#                 c4.markdown(f"**Fail Rate:** {fail_r*100:.1f}%")
#             with b:
#                 st.metric("Score", f"{score:.4f}")

#     st.divider()
#     st.download_button(
#         "⬇️ Export Risk CSV",
#         data=rf[[c for c in ["CustomerID", "Risk_Level", "Risk_Score", "Risk_Reason",
#                               "Current_Balance", "Upcoming_Total_Debit",
#                               "Subscription_Count", "Failed_Debit_Rate"] if c in rf.columns]
#                 ].to_csv(index=False),
#         file_name="risk_report.csv", mime="text/csv"
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 4 — MONTHLY SPEND
# # ══════════════════════════════════════════════════════════════════════════════
# def tab_spend(data, sel_cust, d_from, d_to):
#     ins     = data.get("insights", pd.DataFrame())
#     summary = data.get("summary",  pd.DataFrame())

#     ins_f = ins.copy()
#     if sel_cust != "All Customers" and not ins_f.empty and "CustomerID" in ins_f.columns:
#         ins_f = ins_f[ins_f["CustomerID"] == sel_cust]

#     if ins_f.empty or "Active_Subscriptions" not in ins_f.columns:
#         st.info("Run the pipeline to generate insights."); return

#     k1, k2, k3 = st.columns(3)
#     k1.metric("Avg Subscriptions / Customer", f"{ins_f['Active_Subscriptions'].mean():.1f}")
#     k2.metric("Avg Monthly Spend",            f"Rs{ins_f['Total_Monthly_Spend'].mean():,.2f}")
#     k3.metric("Max Monthly Spend",            f"Rs{ins_f['Total_Monthly_Spend'].max():,.2f}")

#     if sel_cust != "All Customers" and not ins_f.empty:
#         row = ins_f.iloc[0]
#         msg = row.get(
#             "FR8_Message",
#             f"You have {row.get('Active_Subscriptions', 0)} active subscription(s). "
#             f"Total monthly spend: Rs{row.get('Total_Monthly_Spend', 0):,.2f}."
#         )
#         st.info(f"📌 **FR8 Insight — {sel_cust}:** {msg}")
#     else:
#         st.info(
#             f"📌 **FR8 Insight:** Average {ins_f['Active_Subscriptions'].mean():.1f} "
#             f"active subscriptions | Average monthly spend "
#             f"Rs{ins_f['Total_Monthly_Spend'].mean():,.2f}"
#         )

#     st.divider()

#     if sel_cust == "All Customers":
#         cl, cr = st.columns(2)
#         with cl:
#             st.subheader("Monthly Spend Distribution")
#             fig = px.histogram(ins_f, x="Total_Monthly_Spend", nbins=30,
#                                color_discrete_sequence=["#2ecc71"],
#                                labels={"Total_Monthly_Spend": "Monthly Spend (Rs)"})
#             st.plotly_chart(fig, use_container_width=True)
#         with cr:
#             st.subheader("Top 15 Customers by Spend")
#             top15 = ins_f.sort_values("Total_Monthly_Spend", ascending=False).head(15)
#             fig2  = px.bar(top15, x="CustomerID", y="Total_Monthly_Spend",
#                            color="Active_Subscriptions",
#                            labels={"Total_Monthly_Spend": "Monthly Spend (Rs)"})
#             fig2.update_xaxes(tickangle=45)
#             st.plotly_chart(fig2, use_container_width=True)
#     else:
#         # Chart 2 — Spend by Merchant pie (BRD required)
#         cs = summary[summary["CustomerID"] == sel_cust].copy() \
#              if not summary.empty else pd.DataFrame()
#         if not cs.empty:
#             cl, cr = st.columns(2)
#             with cl:
#                 st.subheader(f"Spend by Merchant — {sel_cust}")
#                 fig = px.pie(cs, names="Merchant", values="Avg_Amount",
#                              color_discrete_sequence=px.colors.qualitative.Set2)
#                 st.plotly_chart(fig, use_container_width=True)
#             with cr:
#                 disp = cs[["Description", "Merchant", "Frequency", "Avg_Amount"]].rename(
#                     columns={"Description": "Subscription", "Avg_Amount": "Monthly Spend (Rs)"})
#                 disp.insert(0, "Type", "Subscription")
#                 st.dataframe(disp, use_container_width=True, hide_index=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 5 — GENAI ALERTS (FR7)
# # Alert Validation and Pipeline Run Summary expanders REMOVED
# # ══════════════════════════════════════════════════════════════════════════════
# def tab_genai(data, sel_cust):
#     risk        = data.get("risk", pd.DataFrame())
#     alerts_path = os.path.join(RPT, "alerts.txt")

#     if not os.path.exists(alerts_path) or os.path.getsize(alerts_path) == 0:
#         st.warning("No alerts found.")
#         st.info("Run `python run_pipeline.py` to generate alerts.")
#         return

#     with open(alerts_path, encoding="utf-8") as f:
#         raw_txt = f.read()

#     blocks, buf = [], []
#     for line in raw_txt.splitlines():
#         if line.startswith("=" * 28) and buf:
#             txt = "\n".join(buf).strip()
#             if len(txt) > 60: blocks.append(txt)
#             buf = [line]
#         else:
#             buf.append(line)
#     if buf:
#         txt = "\n".join(buf).strip()
#         if len(txt) > 60: blocks.append(txt)

#     def get_cid(blk):
#         for ln in blk.splitlines():
#             m = re.search(r'CUSTOMER\s*[:\|]\s*(CUST\S+)', ln, re.IGNORECASE)
#             if m: return m.group(1).strip()
#         return None

#     cid_blocks = [(b, get_cid(b)) for b in blocks if get_cid(b)]
#     filtered   = [(b, c) for b, c in cid_blocks
#                   if sel_cust == "All Customers" or c == sel_cust]
#     if not filtered:
#         st.warning(f"No alerts for {sel_cust}. Showing all.")
#         filtered = cid_blocks
#     if not filtered:
#         st.warning("No alert blocks found. Re-run `python run_pipeline.py`.")
#         return

#     st.markdown(f"**{len(filtered)} alert(s) available**")
#     opts   = [f"{c}  |  Alert {i+1}" for i, (_, c) in enumerate(filtered)]
#     chosen = st.selectbox("Select Alert", opts)
#     idx    = opts.index(chosen)
#     blk, blk_cid = filtered[idx]

#     cl, cr = st.columns([2, 1])
#     with cl:
#         st.code(blk, language=None)
#     with cr:
#         st.markdown(f"**👤 Customer:** `{blk_cid}`")
#         if not risk.empty and "CustomerID" in risk.columns:
#             rr = risk[risk["CustomerID"] == blk_cid]
#             if not rr.empty:
#                 r    = rr.iloc[0]
#                 lvl  = str(r.get("Risk_Level", "-"))
#                 icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(lvl, "⚪")
#                 st.markdown(f"**{icon} Risk Level:** {lvl}")
#                 st.markdown(f"**Score:** `{r.get('Risk_Score', '-')}`")
#                 st.markdown(f"**Balance:** Rs{float(r.get('Current_Balance', 0)):,.0f}")
#                 st.markdown(f"**Upcoming:** Rs{float(r.get('Upcoming_Total_Debit', 0)):,.0f}")
#                 st.markdown(f"**Reason:** {r.get('Risk_Reason', '-')}")


# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 6 — ADD NEW CUSTOMER
# # ══════════════════════════════════════════════════════════════════════════════
# def tab_add_customer():
#     st.markdown(
#         "Enter a new customer's transaction details. "
#         "Runs **FR3 → FR4 → FR5 → FR6 → FR7** instantly."
#     )

#     with st.form("new_cust_form"):
#         st.markdown("#### Customer Details")
#         ca, cb = st.columns(2)
#         with ca:
#             nc_id   = st.text_input("CustomerID *",             placeholder="CUST999999")
#             nc_desc = st.text_input("Transaction Description *", placeholder="NETFLIX MONTHLY SUBSCRIPTION")
#             nc_mer  = st.text_input("Merchant Name *",           placeholder="Netflix")
#         with cb:
#             nc_amt  = st.number_input("Amount (Rs) *",  min_value=0.0, value=0.0, step=1.0)
#             nc_bal  = st.number_input("Balance (Rs) *", min_value=0.0, value=0.0, step=100.0)
#             nc_fail = st.slider("Historical Failure Rate", 0.0, 0.3, 0.0, 0.01,
#                                 help="Fraction of past debits that failed (0 = none)")

#         st.markdown("#### Past Transaction Dates *(min 3 required — FR4)*")
#         d1c, d2c, d3c, d4c = st.columns(4)
#         nd1 = d1c.date_input("Date 1 *",          value=date(2025, 1, 15))
#         nd2 = d2c.date_input("Date 2 *",          value=date(2025, 2, 15))
#         nd3 = d3c.date_input("Date 3 *",          value=date(2025, 3, 15))
#         nd4 = d4c.date_input("Date 4 (optional)", value=date(2025, 4, 15))
#         go  = st.form_submit_button("🚀 Analyse & Predict", type="primary",
#                                     use_container_width=True)

#     if go:
#         past = sorted(list({str(nd1), str(nd2), str(nd3), str(nd4)}))
#         errs = validate_inputs(nc_id, nc_desc, nc_mer, nc_amt, nc_bal, past)
#         if errs:
#             for e in errs: st.error(f"❌ {e}")
#             return

#         with st.spinner("Running FR3 → FR4 → FR5 → FR6 → FR7..."):
#             is_sub, conf             = nlp_classify(nc_desc)
#             nxt_dt, nxt_amt, freq   = predict_next(past, nc_amt)
#             r_score, r_level, r_rsn = compute_risk(float(nc_bal), float(nxt_amt),
#                                                     fail_rate=float(nc_fail), sub_count=1)
#             alert_txt               = build_alert(nc_id, nc_mer, nxt_dt, nxt_amt,
#                                                    float(nc_bal), r_score, r_level, r_rsn)
#         st.success("✅ Analysis complete!")
#         st.divider()

#         m1, m2, m3, m4 = st.columns(4)
#         m1.metric("NLP Detection",    "Subscription ✅" if is_sub else "Non-Sub ❌",
#                   f"{conf*100:.0f}% confidence")
#         m2.metric("Next Debit Date",  str(nxt_dt))
#         m3.metric("Predicted Amount", f"Rs{nxt_amt:,.2f}")
#         icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(r_level, "⚪")
#         m4.metric(f"{icon} Risk Level", r_level, f"Score: {r_score}")

#         cl, cr = st.columns(2)
#         with cl:
#             st.markdown("#### FR3 – FR5: Detection & Prediction")
#             st.dataframe(pd.DataFrame({
#                 "Field": ["CustomerID", "Description", "Merchant", "NLP Result",
#                           "Confidence", "Frequency", "Next Debit Date", "Predicted Amount"],
#                 "Value": [nc_id, nc_desc, nc_mer,
#                           "Subscription ✅" if is_sub else "Non-Subscription ❌",
#                           f"{conf*100:.0f}%", freq, str(nxt_dt), f"Rs{nxt_amt:,.2f}"],
#             }), hide_index=True, use_container_width=True)
#         with cr:
#             st.markdown("#### FR6: Risk Assessment")
#             days_l = (pd.Timestamp(str(nxt_dt)).date() - date.today()).days
#             st.dataframe(pd.DataFrame({
#                 "Field": ["Risk Score", "Risk Level", "Reason", "Balance",
#                           "Upcoming Debit", "Failure Rate", "Days Until Due"],
#                 "Value": [str(r_score), r_level, r_rsn, f"Rs{nc_bal:,.2f}",
#                           f"Rs{nxt_amt:,.2f}", f"{nc_fail*100:.1f}%",
#                           f"{days_l}d" if days_l >= 0 else f"Overdue {abs(days_l)}d"],
#             }), hide_index=True, use_container_width=True)

#         st.markdown("#### FR7: GenAI Alert")
#         st.code(alert_txt, language=None)

#         st.markdown("#### Transaction Timeline")
#         dts = [pd.Timestamp(str(d)) for d in past]
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=dts, y=[float(nc_amt)] * len(dts),
#             mode="lines+markers", name="Historical",
#             line=dict(color="#3498db", width=2), marker=dict(size=10)))
#         fig.add_trace(go.Scatter(
#             x=[pd.Timestamp(str(nxt_dt))], y=[nxt_amt],
#             mode="markers+text", name="Predicted",
#             marker=dict(size=14, color="#e74c3c", symbol="star"),
#             text=[f"Rs{nxt_amt:.0f} (predicted)"], textposition="top center"))
#         fig.update_layout(xaxis_title="Date", yaxis_title="Amount (Rs)",
#                           title=f"Transaction Timeline — {nc_desc}")
#         st.plotly_chart(fig, use_container_width=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 7 — FR10 AI CHATBOT
# # ══════════════════════════════════════════════════════════════════════════════
# def tab_chatbot(data):
#     """
#     GenAI RAG Chatbot — answers questions grounded in CSV data.
#     Uses Phi-2 (D:/phi2_local) if available, else fast rule-based fallback.
#     All answers sourced from actual data — no hallucinations.
#     """
#     if "chat_history" not in st.session_state:
#         st.session_state["chat_history"] = []
#     if "pending_question" not in st.session_state:
#         st.session_state["pending_question"] = None

#     # ── Header row ──
#     hc1, hc2 = st.columns([3, 1])
#     with hc1:
#         st.markdown("#### 🤖 SubIntel AI Chatbot")
#         model_label = "Microsoft Phi-2 (local)" if _CHATBOT_READY else "Rule-Based Fallback (fast)"
#         st.caption(
#             f"Model: **{model_label}** · "
#             "Answers grounded in your data · No hallucinations · "
#             "Target latency < 3 sec"
#         )
#     with hc2:
#         risk = data.get("risk", pd.DataFrame())
#         cust_opts = ["All Customers"]
#         if not risk.empty and "CustomerID" in risk.columns:
#             cust_opts += sorted(risk["CustomerID"].unique().tolist())
#         chat_cid = st.selectbox(
#             "👤 Customer context",
#             cust_opts,
#             key="chatbot_cid_select",
#             help="Select a customer for context-aware answers, or keep 'All Customers' for global stats"
#         )

#     st.divider()

#     # ── Example question buttons ──
#     with st.expander("💡 Example questions — click any to ask", expanded=True):
#         examples = [
#             "How many active subscriptions do I have?",
#             "What is my total monthly subscription spend?",
#             "Which subscriptions are at risk of failure?",
#             "Which subscriptions am I not using?",
#             "When is my next debit?",
#             "Who are the top merchants?",
#             "Give me a summary",
#             "Show me high risk customers",
#             "What is my account balance?",
#         ]
#         cols = st.columns(3)
#         for i, ex in enumerate(examples):
#             with cols[i % 3]:
#                 if st.button(ex, key=f"ex_{i}", use_container_width=True):
#                     st.session_state["pending_question"] = ex

#     st.divider()

#     # ── Chat history display ──
#     chat_container = st.container(height=400, border=True)
#     with chat_container:
#         if not st.session_state["chat_history"]:
#             st.markdown(
#                 "<div style='text-align:center; color:#888; padding:60px 0;'>"
#                 "👋 Hi! I'm SubIntel AI. Ask me about subscriptions, risk, debits, or spend.<br>"
#                 "<small>Try clicking one of the example questions above to get started.</small>"
#                 "</div>",
#                 unsafe_allow_html=True,
#             )
#         for msg in st.session_state["chat_history"]:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])
#                 if msg["role"] == "assistant" and "meta" in msg:
#                     meta = msg["meta"]
#                     st.caption(
#                         f"🎯 Intent: `{meta.get('intent', '?')}` · "
#                         f"🤖 Model: `{meta.get('model', '?')}` · "
#                         f"⚡ Latency: `{meta.get('latency_ms', '?')} ms`"
#                     )

#     # ── Chat input ──
#     pending = st.session_state.pop("pending_question", None)
#     user_input = st.chat_input(
#         "Ask about subscriptions, risk, upcoming debits, monthly spend...",
#         key="chatbot_input"
#     )
#     query = pending or user_input

#     if query:
#         st.session_state["chat_history"].append({"role": "user", "content": query})
#         cid = None if chat_cid == "All Customers" else chat_cid
#         with st.spinner("Analysing..."):
#             result = chat(query, data, customer_id=cid)
#         st.session_state["chat_history"].append({
#             "role":    "assistant",
#             "content": result["answer"],
#             "meta":    result,
#         })
#         st.rerun()

#     # ── Clear chat ──
#     if st.session_state["chat_history"]:
#         if st.button("🗑️ Clear conversation", key="clear_chat"):
#             st.session_state["chat_history"] = []
#             st.rerun()

#     # ── Performance panel ──
#     with st.expander("📊 Chatbot Performance & Architecture", expanded=False):
#         msgs = st.session_state["chat_history"]
#         assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
#         latencies = [m["meta"]["latency_ms"] for m in assistant_msgs if "meta" in m]
#         avg_lat = int(sum(latencies) / len(latencies)) if latencies else 0

#         mc1, mc2, mc3, mc4 = st.columns(4)
#         mc1.metric("Responses given",  len(assistant_msgs))
#         mc2.metric("Avg latency",      f"{avg_lat} ms",
#                    delta="✅ Under target" if avg_lat < 3000 else "⚠️ Over 3s",
#                    delta_color="normal")
#         mc3.metric("Model",            "Phi-2 (local)" if _CHATBOT_READY else "Rule-based")
#         mc4.metric("Hallucinations",   "0 ✅", help="All answers grounded in CSV data")

#         st.markdown("""
# **RAG Pipeline Architecture:**
# ```
# User Query
#     ↓
# Intent Classifier  (keyword pattern match — instant)
#     ↓
# Context Retriever  (pulls relevant rows from CSV data — the "RAG" step)
#     ↓
# Phi-2 / Rule-based answerer  (grounded prompt → answer)
#     ↓
# Response  (with intent, model, latency shown)
# ```
# **Validated question types (BRD compliant):**

# | Question | Answer type |
# |---|---|
# | Active subscription count | Count + merchant list |
# | Total monthly spend | Amount breakdown |
# | At-risk subscriptions | Risk level + score + reason |
# | Unused subscriptions | Low-occurrence list |
# | Next debit date | Date + amount predicted |
# | Top merchants | Ranked list |
# | Account balance | Balance vs upcoming debit |
# | High-risk customer list | Top 5 with reasons |
#         """)


# # ══════════════════════════════════════════════════════════════════════════════
# # ABOUT PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# def about_page():
#     st.markdown("## ℹ️ About SubIntel")
#     st.divider()
#     with st.container(border=True):
#         st.markdown("### 🏦 System Overview")
#         st.markdown(
#             "The **AI Subscription & Auto-Debit Intelligence System** helps banks proactively "
#             "identify subscription transactions, predict upcoming debits, assess failure risk, "
#             "and generate smart alerts — powered by NLP and machine learning."
#         )
#     with st.container(border=True):
#         st.markdown("### ⚙️ Technology Stack")
#         st.markdown("""
# | FR | Component | Technology |
# |---|---|---|
# | FR3 | NLP Detection | spaCy + TF-IDF + Logistic Regression |
# | FR4 | Pattern Detection | Date gap analysis (median gap) |
# | FR5 | Debit Prediction | ARIMA(1,0,0) + Linear Regression blend |
# | FR6 | Risk Scoring | Gradient Boosting Classifier |
# | FR7 | GenAI Alerts | Microsoft Phi-2 (rule-based fallback) |
# | FR8 | Insights | Customer monthly spend summary |
# | FR9 | Dashboard | Streamlit + Plotly |
# | FR1 | Data | Faker synthetic generator — 134K+ transactions |
# """)
#     with st.container(border=True):
#         st.markdown("### 📋 BRD Compliance (FR1–FR9)")
#         for fr, desc in [
#             ("FR1", "Synthetic dataset — 134K+ transactions (Faker)"),
#             ("FR2", "Data cleaning — null handling, text normalisation"),
#             ("FR3", "NLP subscription detection — >90% accuracy (spaCy + TF-IDF + LR)"),
#             ("FR4", "Recurring pattern detection — min 3 occurrences, gap-based frequency"),
#             ("FR5", "Next debit prediction — ARIMA(1,0,0) + Linear Regression"),
#             ("FR6", "Risk scoring — Gradient Boosting, >85% accuracy, explainable reasons"),
#             ("FR7", "GenAI alerts — Phi-2 with structured rule-based fallback"),
#             ("FR8", "Customer insights — monthly spend, active subscriptions"),
#             ("FR9", "Interactive Streamlit dashboard with Plotly charts"),
#         ]:
#             st.markdown(f"✅ **{fr}:** {desc}")
#     with st.container(border=True):
#         st.markdown("### 👥 Team 7")
#         st.markdown("**Mansi & Samyak** — AI Subscription & Auto-Debit Intelligence System")


# # ══════════════════════════════════════════════════════════════════════════════
# # HOME PAGE
# # ══════════════════════════════════════════════════════════════════════════════
# def home_page(data):
#     st.markdown("## 🏦 SubIntel — AI Subscription Intelligence Dashboard")
#     st.divider()

#     sel_cust, sel_merch, d_from, d_to = render_filters(data)
#     st.divider()
#     render_kpis(data, sel_cust, d_from, d_to)
#     st.divider()

#     tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
#         "🔄 Active Subscriptions",
#         "📅 Upcoming Debits",
#         "⚠️ Risk Alerts",
#         "📈 Monthly Spend",
#         "🤖 GenAI Alerts",
#         "➕ Add New Customer",
#         "💬 AI Chatbot",
#     ])
#     with tab1: tab_subscriptions(data, sel_cust, sel_merch, d_from, d_to)
#     with tab2: tab_upcoming(data, sel_cust, sel_merch, d_from, d_to)
#     with tab3: tab_risk(data, sel_cust, sel_merch)
#     with tab4: tab_spend(data, sel_cust, d_from, d_to)
#     with tab5: tab_genai(data, sel_cust)
#     with tab6: tab_add_customer()
#     with tab7: tab_chatbot(data)

#     st.divider()
#     st.caption("© 2025 SubIntel — Team 7 (Mansi & Samyak) | AI Subscription & Auto-Debit Intelligence System")

# # def home_page(data):
# #     st.markdown("## 🏦 SubIntel — AI Subscription Intelligence Dashboard")
# #     st.divider()

# #     sel_cust, sel_merch, d_from, d_to = render_filters(data)
# #     st.divider()
# #     render_kpis(data, sel_cust, d_from, d_to)
# #     st.divider()

# #     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
# #         "🔄 Active Subscriptions",
# #         "📅 Upcoming Debits",
# #         "⚠️ Risk Alerts",
# #         "📈 Monthly Spend",
# #         "🤖 GenAI Alerts",
# #         "➕ Add New Customer",
# #     ])
# #     with tab1: tab_subscriptions(data, sel_cust, sel_merch, d_from, d_to)
# #     with tab2: tab_upcoming(data, sel_cust, sel_merch, d_from, d_to)
# #     with tab3: tab_risk(data, sel_cust, sel_merch)
# #     with tab4: tab_spend(data, sel_cust, d_from, d_to)
# #     with tab5: tab_genai(data, sel_cust)
# #     with tab6: tab_add_customer()

# #     st.divider()
# #     st.caption("© 2025 SubIntel — Team 7 (Mansi & Samyak) | AI Subscription & Auto-Debit Intelligence System")


# # ══════════════════════════════════════════════════════════════════════════════
# # STATIC MATPLOTLIB REPORT  (called by run_pipeline.py for PNG report)
# # ══════════════════════════════════════════════════════════════════════════════
# def generate_static_dashboard(df, pred_df, risk_df, summary_df,
#                                insights_df, output_path, nlp_metrics=None):
#     import matplotlib; matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
#     from matplotlib.patches import FancyBboxPatch

#     P = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71",
#          "blue": "#2980b9", "dark": "#2c3e50"}
#     fig = plt.figure(figsize=(24, 18), facecolor="white")
#     fig.suptitle("AI Subscription & Auto-Debit Intelligence System\n"
#                  "Team 7 - Mansi & Samyak | spaCy + ARIMA + Gradient Boosting + Phi-2",
#                  fontsize=15, fontweight="bold", color=P["dark"], y=0.99)
#     gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.58, wspace=0.42)

#     sub_pred = df["NLP_Sub_Pred"].sum() if "NLP_Sub_Pred" in df.columns \
#                else df.get("SubscriptionFlag", pd.Series([0])).sum()
#     high_cnt = (risk_df["Risk_Level"] == "High").sum() if not risk_df.empty else 0

#     for col, (title, val, sub, clr) in enumerate([
#         ("Total Transactions",  f"{len(df):,}",            "",           P["blue"]),
#         ("Subscriptions",       f"{int(sub_pred):,}",      f"{sub_pred/len(df)*100:.1f}%", "#8e44ad"),
#         ("Confirmed Recurring", f"{int(df['Is_Recurring'].sum()):,}", ">=3 occ", "#16a085"),
#         ("High Risk",           f"{int(high_cnt):,}",      f"of {len(risk_df):,}", P["High"]),
#     ]):
#         ax = fig.add_subplot(gs[0, col])
#         ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
#         ax.add_patch(FancyBboxPatch((0.05, 0.05), 0.90, 0.90, boxstyle="round,pad=0.02",
#                                     lw=2, edgecolor=clr, facecolor=clr + "25"))
#         ax.text(0.5, 0.72, title, ha="center", fontsize=8.5, color=P["dark"], fontweight="bold")
#         ax.text(0.5, 0.44, val,   ha="center", fontsize=18,  color=clr,       fontweight="bold")
#         if sub: ax.text(0.5, 0.20, sub, ha="center", fontsize=7, color="grey")

#     ax1 = fig.add_subplot(gs[1, 0:2])
#     df["Date"] = pd.to_datetime(df["Date"])
#     m = df.groupby(df["Date"].dt.to_period("M")).size().reset_index()
#     m.columns = ["P", "C"]; m["P"] = m["P"].astype(str)
#     ax1.fill_between(range(len(m)), m["C"], alpha=0.3, color=P["blue"])
#     ax1.plot(range(len(m)), m["C"], color=P["blue"], lw=2)
#     ax1.set_xticks(range(0, len(m), 3))
#     ax1.set_xticklabels(m["P"].iloc[::3], rotation=45, fontsize=7)
#     ax1.set_title("Transaction Volume", fontweight="bold", fontsize=10)
#     ax1.set_ylabel("Count"); ax1.grid(axis="y", alpha=0.3)
#     ax1.spines[["top", "right"]].set_visible(False)

#     ax2 = fig.add_subplot(gs[1, 2])
#     if not risk_df.empty and "Risk_Level" in risk_df.columns:
#         rc = risk_df["Risk_Level"].value_counts()
#         ax2.pie(rc.values, labels=rc.index,
#                 colors=[P.get(l, "#aaa") for l in rc.index],
#                 autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
#     ax2.set_title("Risk Distribution", fontweight="bold", fontsize=10)

#     ax3 = fig.add_subplot(gs[1, 3])
#     if nlp_metrics:
#         keys = ["accuracy", "precision", "recall", "f1"]
#         vals = [nlp_metrics.get(k, 0) for k in keys]
#         bars = ax3.bar(["Acc", "Prec", "Rec", "F1"], vals,
#                        color=[P["blue"] if v >= 0.90 else P["Medium"] for v in vals],
#                        edgecolor="white")
#         ax3.axhline(0.90, color=P["High"], ls="--", lw=1.2)
#         ax3.set_ylim(0, 1.12)
#         ax3.set_title("NLP Metrics (FR3)", fontweight="bold", fontsize=10)
#         for bar, v in zip(bars, vals):
#             ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
#                      f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
#         ax3.spines[["top", "right"]].set_visible(False)

#     ax4 = fig.add_subplot(gs[2, 0:2])
#     if not summary_df.empty:
#         ts = summary_df.groupby("Description")["Occurrences"].sum() \
#                        .sort_values(ascending=False).head(10)
#         ax4.barh([d[:32] for d in ts.index][::-1], ts.values[::-1],
#                  color=P["blue"], edgecolor="white")
#         ax4.set_title("Top 10 Subscriptions", fontweight="bold", fontsize=10)
#         ax4.set_xlabel("Occurrences")
#         ax4.spines[["top", "right"]].set_visible(False)

#     ax5 = fig.add_subplot(gs[2, 2])
#     if "Is_Recurring" in df.columns and "Inferred_Freq" in df.columns:
#         fd = df[df["Is_Recurring"] == 1]["Inferred_Freq"].value_counts()
#         if not fd.empty:
#             ax5.bar(fd.index, fd.values,
#                     color=["#16a085", "#8e44ad"][:len(fd)], edgecolor="white")
#     ax5.set_title("Recurring Frequency (FR4)", fontweight="bold", fontsize=10)
#     ax5.set_ylabel("Count")
#     ax5.spines[["top", "right"]].set_visible(False)

#     ax6 = fig.add_subplot(gs[2, 3])
#     if not risk_df.empty and "Risk_Score" in risk_df.columns:
#         ax6.hist(risk_df["Risk_Score"], bins=30, color=P["blue"], edgecolor="white", alpha=0.8)
#         ax6.axvline(0.35, color=P["Medium"], ls="--", lw=1.5, label="Med")
#         ax6.axvline(0.65, color=P["High"],   ls="--", lw=1.5, label="High")
#         ax6.set_title("Risk Score Distribution (FR6)", fontweight="bold", fontsize=10)
#         ax6.set_xlabel("Score"); ax6.set_ylabel("Accounts"); ax6.legend(fontsize=7)
#         ax6.spines[["top", "right"]].set_visible(False)

#     ax7 = fig.add_subplot(gs[3, :])
#     ax7.axis("off")
#     if not risk_df.empty:
#         tr = risk_df.sort_values("Risk_Score", ascending=False).head(8)[
#             ["CustomerID", "Current_Balance", "Upcoming_Total_Debit",
#              "Failed_Debit_Rate", "Subscription_Count", "Risk_Score", "Risk_Level"]].copy()
#         tr["Current_Balance"]      = tr["Current_Balance"].map("Rs{:,.0f}".format)
#         tr["Upcoming_Total_Debit"] = tr["Upcoming_Total_Debit"].map("Rs{:,.0f}".format)
#         tr["Failed_Debit_Rate"]    = tr["Failed_Debit_Rate"].map("{:.1%}".format)
#         tr["Risk_Score"]           = tr["Risk_Score"].map("{:.4f}".format)
#         tr["Subscription_Count"]   = tr["Subscription_Count"].astype(int)
#         tbl = ax7.table(
#             cellText=tr.values,
#             colLabels=["CustomerID", "Balance", "Upcoming", "Fail%", "Subs", "Score", "Level"],
#             cellLoc="center", loc="center", bbox=[0, 0.05, 1, 0.90]
#         )
#         tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
#         for (row, col), cell in tbl.get_celld().items():
#             if row == 0:
#                 cell.set_facecolor(P["dark"])
#                 cell.set_text_props(color="white", fontweight="bold")
#             elif row % 2 == 0:
#                 cell.set_facecolor("#f5f5f5")
#         ax7.set_title("Top High-Risk Accounts", fontweight="bold", fontsize=10, pad=8)

#     plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
#     print(f"[Dashboard] Saved -> {output_path}")
#     plt.close()


# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════════════════════
# def run_dashboard():
#     st.set_page_config(
#         page_title="SubIntel - AI Subscription Intelligence",
#         page_icon="🏦",
#         layout="wide",
#     )
#     if "logged_in" not in st.session_state:
#         st.session_state["logged_in"] = False
#     if "page" not in st.session_state:
#         st.session_state["page"] = "Home"

#     if not st.session_state["logged_in"]:
#         login_page()
#         return

#     render_sidebar()
#     data = load_data()

#     if st.session_state.get("page") == "About":
#         about_page()
#     else:
#         home_page(data)


# if __name__ == "__main__":
#     run_dashboard()

####################################################################################

"""
FR9 - Streamlit Dashboard
AI Subscription & Auto-Debit Intelligence System
Team 7 - Mansi & Samyak
 
Tabs: Active Subscriptions | Upcoming Debits | Risk Alerts | Monthly Spend | GenAI Alerts | Add New Customer
Run: streamlit run streamlit_app/dashboard.py
"""
 
import os, re, json
from datetime import date, timedelta
 
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
 
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RPT  = os.path.join(BASE, "reports")
 
USERS       = {"banker01": "pass123", "banker02": "pass456", "admin": "admin123"}
RISK_COLORS = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
 
SUB_WORDS = {
    "netflix","spotify","amazon","youtube","hotstar","apple","google","microsoft",
    "linkedin","dropbox","adobe","zoom","coursera","swiggy","subscription","premium",
    "membership","monthly","weekly","annual","renewal","electricity","utility",
    "prime","recurring","gym","plan","zee5","bescom","tatasky","jiocinema",
}
FP_WORDS = {"salary","wages","refund","dividend","interest","deposit","bonus","neft","imps","rtgs","credit"}
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading data...")
def load_data():
    files = {
        "raw":      "transactions_raw.csv",
        "patterns": "transactions_patterns.csv",
        "pred":     "predictions.csv",
        "risk":     "risk_scored.csv",
        "summary":  "recurring_summary.csv",
        "insights": "customer_insights.csv",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(DATA, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for col in ["Date", "Last_Date", "Next_Debit_Date", "Last_Debit_Date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            data[key] = df
        else:
            data[key] = pd.DataFrame()
    return data
 
 
# ══════════════════════════════════════════════════════════════════════════════
# INLINE PIPELINE HELPERS (Add New Customer)
# ══════════════════════════════════════════════════════════════════════════════
def nlp_classify(text):
    tokens = set(re.sub(r"[^a-z\s]", " ", text.lower()).split())
    if tokens & FP_WORDS:  return False, 0.02
    if tokens & SUB_WORDS: return True,  0.97
    return False, 0.40
 
 
def predict_next(date_strs, amount):
    dates = sorted(pd.Timestamp(str(d)) for d in date_strs)
    gaps  = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    gap   = int(np.median(gaps))
    freq  = "Weekly" if gap <= 9 else ("Monthly" if gap <= 35 else "Irregular")
    return (dates[-1] + timedelta(days=gap)).date(), round(float(amount), 2), freq
 
 
def compute_risk(balance, upcoming, fail_rate=0.0, sub_count=1):
    p_fail   = min(fail_rate / max(0.07, fail_rate + 0.001), 1.0)
    p_subs   = min(sub_count / 4.0, 1.0)
    p_upco   = min(upcoming / max(upcoming * 1.5, 1), 1.0)
    p_burden = min((upcoming / max(balance, 1)) / 0.05, 1.0)
    score    = round(float(np.clip(
        0.35 * p_fail + 0.30 * p_subs + 0.20 * p_upco + 0.15 * p_burden, 0, 1)), 4)
    level    = "High" if score >= 0.65 else ("Medium" if score >= 0.35 else "Low")
    reasons  = []
    if fail_rate > 0.05:
        reasons.append(f"Failure rate {fail_rate*100:.1f}%")
    if sub_count >= 3:
        reasons.append(f"{sub_count} active subscriptions")
    if upcoming > 0:
        upco_pct = (upcoming / max(balance, 1)) * 100
        reasons.append(f"Upcoming debit Rs{upcoming:,.0f} ({upco_pct:.1f}% of balance)")
    reason = "; ".join(reasons) if reasons else f"{sub_count} subscription(s), Rs{upcoming:,.0f} upcoming"
    return score, level, reason
 
 
def build_alert(cid, merchant, next_date, next_amt, balance, score, level, reason):
    days      = (pd.Timestamp(str(next_date)).date() - date.today()).days
    timing    = f"in {days} day(s)" if days > 0 else "TODAY" if days == 0 else f"overdue {abs(days)}d"
    shortfall = max(0, next_amt - balance)
    if level == "High":
        sug = f"Top up Rs{shortfall:,.0f} immediately. Debit due {timing}."
    elif level == "Medium":
        sug = f"Maintain Rs{next_amt*1.1:,.0f} before {next_date}."
    else:
        sug = "Account balance sufficient. No action needed."
    return (
        f"{'='*56}\n  CUSTOMER: {cid}\n{'='*56}\n"
        f"\n[FR7] ALERT\n"
        f"  Upcoming Debit: {merchant} Rs{next_amt:,.2f} | "
        f"Risk: {level} ({score}) | Suggestion: {sug}\n"
        f"\n[FR6] RISK ASSESSMENT\n"
        f"  Score  : {score} | Level: {level}\n"
        f"  Reason : {reason}\n"
        f"  Balance: Rs{balance:,.2f} | Upcoming: Rs{next_amt:,.2f}\n"
        f"\n[FR8] INSIGHT\n"
        f"  1 active subscription | Monthly spend: Rs{next_amt:,.2f}\n"
        f"{'='*56}"
    )
 
 
def validate_inputs(cid, desc, merchant, amt, bal, dates):
    errs = []
    if not re.match(r"^[A-Za-z0-9_\-]{3,20}$", cid.strip()):
        errs.append("CustomerID must be 3-20 alphanumeric characters.")
    if len(desc.strip()) < 3:
        errs.append("Description must be at least 3 characters.")
    if not merchant.strip():
        errs.append("Merchant Name is required.")
    if amt <= 0:
        errs.append("Amount must be > Rs0.")
    if bal < 0:
        errs.append("Balance cannot be negative.")
    if len(dates) < 3:
        errs.append(f"Need at least 3 past dates (BRD FR4). Provided: {len(dates)}.")
    return errs
 
 
# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("### 🏦 AI Subscription Intelligence")
        st.markdown("**SubIntel — Banker Portal**")
        st.divider()
        with st.form("login_form"):
            uid = st.text_input("Banker ID", placeholder="banker01")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                if uid in USERS and USERS[uid] == pwd:
                    st.session_state.update(logged_in=True, banker_id=uid, page="Home")
                    st.rerun()
                else:
                    st.error("Invalid Banker ID or Password.")
        st.caption("Demo credentials: banker01 / pass123")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏦 SubIntel")
        st.markdown(f"👤 **{st.session_state.get('banker_id', '')}**")
        st.divider()
        if st.button("🏠 Home",     use_container_width=True):
            st.session_state["page"] = "Home";  st.rerun()
        if st.button("ℹ️ About",    use_container_width=True):
            st.session_state["page"] = "About"; st.rerun()
        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            for k in ["logged_in", "banker_id", "page"]:
                st.session_state.pop(k, None)
            st.rerun()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FILTERS
# Single unified dropdown (Customer or Merchant) + date range tuple input
# ══════════════════════════════════════════════════════════════════════════════
def render_filters(data):
    raw     = data.get("raw",     pd.DataFrame())
    summary = data.get("summary", pd.DataFrame())
    risk    = data.get("risk",    pd.DataFrame())
 
    # Customer options
    cust_options = ["All Customers"]
    if not risk.empty and "CustomerID" in risk.columns:
        cust_options += sorted(risk["CustomerID"].unique().tolist())
 
    # Merchant options
    merch_options = ["All Merchants"]
    if not summary.empty and "Merchant" in summary.columns:
        merch_options += sorted(summary["Merchant"].dropna().unique().tolist())
 
    # Actual data date range
    min_d = max_d = date.today()
    if not raw.empty and "Date" in raw.columns:
        vd = raw["Date"].dropna()
        if not vd.empty:
            min_d, max_d = vd.min().date(), vd.max().date()
 
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.markdown("👤 **Customer**")
        sel_cust = st.selectbox(
            "Customer",
            cust_options,
            label_visibility="collapsed",
            help="Filter by a specific customer"
        )
    with c2:
        st.markdown("🏪 **Merchant**")
        sel_merch = st.selectbox(
            "Merchant",
            merch_options,
            label_visibility="collapsed",
            help="Filter by a specific merchant"
        )
    with c3:
        st.markdown("📅 **Date Range**")
        date_range = st.date_input(
            "Date Range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            label_visibility="collapsed",
        )
 
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_from, d_to = date_range[0], date_range[1]
    else:
        d_from, d_to = min_d, max_d
 
    return sel_cust, sel_merch, d_from, d_to
 
# def render_filters(data):
#     raw     = data.get("raw",     pd.DataFrame())
#     summary = data.get("summary", pd.DataFrame())
#     risk    = data.get("risk",    pd.DataFrame())
 
#     # Build one unified list: All Data, Customer: X, Merchant: Y
#     options = ["All Data"]
#     if not risk.empty and "CustomerID" in risk.columns:
#         options += [f"Customer: {c}" for c in sorted(risk["CustomerID"].unique().tolist())]
#     if not summary.empty and "Merchant" in summary.columns:
#         options += [f"Merchant: {m}" for m in sorted(summary["Merchant"].dropna().unique().tolist())]
 
#     # Actual data date range
#     min_d = max_d = date.today()
#     if not raw.empty and "Date" in raw.columns:
#         vd = raw["Date"].dropna()
#         if not vd.empty:
#             min_d, max_d = vd.min().date(), vd.max().date()
 
#     c1, c2 = st.columns([2, 2])
#     with c1:
#         chosen = st.selectbox(
#             "🔍 Filter by Customer or Merchant",
#             options,
#             help="Choose a specific customer or merchant, or keep 'All Data' to see everything"
#         )
#     with c2:
#         # Tuple date_input avoids the from > to bug — user picks both ends at once
#         date_range = st.date_input(
#             "📅 Date Range",
#             value=(min_d, max_d),
#             min_value=min_d,
#             max_value=max_d,
#         )
 
#     # date_input returns a tuple once both ends are selected, else a single date
#     if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
#         d_from, d_to = date_range[0], date_range[1]
#     else:
#         # User is mid-selection; hold previous values
#         d_from, d_to = min_d, max_d
 
#     # Decode the selection
#     if chosen == "All Data":
#         sel_cust, sel_merch = "All Customers", "All Merchants"
#     elif chosen.startswith("Customer: "):
#         sel_cust  = chosen[len("Customer: "):]
#         sel_merch = "All Merchants"
#     elif chosen.startswith("Merchant: "):
#         sel_cust  = "All Customers"
#         sel_merch = chosen[len("Merchant: "):]
#     else:
#         sel_cust, sel_merch = "All Customers", "All Merchants"
 
#     return sel_cust, sel_merch, d_from, d_to
 
 
# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
def render_kpis(data, sel_cust, d_from, d_to):
    raw  = data.get("raw",  pd.DataFrame())
    risk = data.get("risk", pd.DataFrame())
    pred = data.get("pred", pd.DataFrame())
 
    rf = raw.copy()
    if not rf.empty and "Date" in rf.columns:
        rf = rf[(rf["Date"].dt.date >= d_from) & (rf["Date"].dt.date <= d_to)]
    if sel_cust != "All Customers" and "CustomerID" in rf.columns:
        rf = rf[rf["CustomerID"] == sel_cust]
 
    txn  = len(rf)
    subs = int(rf["SubscriptionFlag"].sum()) if "SubscriptionFlag" in rf.columns else 0
    high = int((risk["Risk_Level"] == "High").sum())   if not risk.empty and "Risk_Level" in risk.columns else 0
    med  = int((risk["Risk_Level"] == "Medium").sum()) if not risk.empty and "Risk_Level" in risk.columns else 0
    upco = float(pred["Predicted_Amount"].sum())       if not pred.empty and "Predicted_Amount" in pred.columns else 0
 
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Transactions",  f"{txn:,}")
    k2.metric("Subscriptions Found", f"{subs:,}")
    k3.metric("🔴 High Risk",        f"{high:,}")
    k4.metric("🟡 Medium Risk",      f"{med:,}")
    k5.metric("💰 Upcoming Debits",  f"Rs{upco:,.0f}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# GAUGE CHART HELPER  (used in Risk Alerts for single-customer/merchant view)
# ══════════════════════════════════════════════════════════════════════════════
def render_gauge(score, level, title="Risk Score"):
    color = RISK_COLORS.get(level, "#aaa")
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(score, 4),
        title = {"text": title, "font": {"size": 16}},
        delta = {"reference": 0.35,
                 "increasing": {"color": "#e74c3c"},
                 "decreasing": {"color": "#2ecc71"}},
        gauge = {
            "axis":  {"range": [0, 1], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,    0.35], "color": "#0d2118"},
                {"range": [0.35, 0.65], "color": "#2d2108"},
                {"range": [0.65, 1.0],  "color": "#2d1012"},
            ],
            "threshold": {
                "line":      {"color": color, "width": 4},
                "thickness": 0.75,
                "value":     score,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ACTIVE SUBSCRIPTIONS
# ══════════════════════════════════════════════════════════════════════════════
def tab_subscriptions(data, sel_cust, sel_merch, d_from, d_to):
    summary = data.get("summary", pd.DataFrame())
    if summary.empty:
        st.info("Run `python run_pipeline.py` to generate subscription data."); return
 
    sf = summary.copy()
    if "Last_Date" in sf.columns:
        sf = sf[(sf["Last_Date"].dt.date >= d_from) & (sf["Last_Date"].dt.date <= d_to)]
    if sel_merch != "All Merchants" and "Merchant" in sf.columns:
        sf = sf[sf["Merchant"] == sel_merch]
    if sel_cust != "All Customers" and "CustomerID" in sf.columns:
        sf = sf[sf["CustomerID"] == sel_cust]
 
    if sf.empty:
        st.warning("No subscriptions match current filters."); return
 
    # BRD-required metrics only — clean single row
    k1, k2, k3 = st.columns(3)
    k1.metric("Recurring Groups",  f"{len(sf):,}")
    k2.metric("Unique Customers",  f"{sf['CustomerID'].nunique():,}")
    k3.metric("Avg Monthly Spend", f"Rs{sf['Avg_Amount'].mean():,.0f}" if "Avg_Amount" in sf.columns else "-")
 
    st.divider()
 
    sort_col = st.selectbox("Sort by", ["Avg_Amount", "Occurrences", "Frequency", "Median_Gap_Days"])
    sf = sf.sort_values(sort_col, ascending=False)
 
    cols = [c for c in ["CustomerID", "Description", "Merchant", "Frequency",
                         "Occurrences", "Avg_Amount", "Median_Gap_Days", "Failed_Count"] if c in sf.columns]
    st.dataframe(
        sf[cols].rename(columns={
            "Avg_Amount":      "Avg Amount (Rs)",
            "Median_Gap_Days": "Gap (days)",
            "Failed_Count":    "Failures",
        }),
        width='stretch', hide_index=True
    )
    st.download_button("⬇️ Export CSV", data=sf[cols].to_csv(index=False),
                       file_name="active_subscriptions.csv", mime="text/csv")
 
    # Chart 1 — Top subscriptions bar chart (BRD required)
    st.subheader("Top Subscriptions by Occurrence")
    top = sf.groupby("Description")["Occurrences"].sum() \
            .sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top, x="Occurrences", y="Description", orientation="h",
                 color="Occurrences", color_continuous_scale="Blues",
                 labels={"Description": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPCOMING DEBITS
# ══════════════════════════════════════════════════════════════════════════════
def tab_upcoming(data, sel_cust, sel_merch, d_from, d_to):
    pred = data.get("pred", pd.DataFrame())
    if pred.empty:
        st.info("Run the pipeline to generate predictions."); return
 
    pf = pred.copy()
    if "Next_Debit_Date" in pf.columns:
        next30 = date.today() + timedelta(days=30)
        pf = pf[(pf["Next_Debit_Date"].dt.date >= d_from) &
                (pf["Next_Debit_Date"].dt.date <= next30)]
    if sel_merch != "All Merchants" and "Merchant" in pf.columns:
        pf = pf[pf["Merchant"] == sel_merch]
    if sel_cust != "All Customers" and "CustomerID" in pf.columns:
        pf = pf[pf["CustomerID"] == sel_cust]
 
    if pf.empty:
        st.warning("No upcoming debits in the next 30 days for current filters."); return
 
    sort_col = st.selectbox("Sort by", ["Next_Debit_Date", "Predicted_Amount", "Frequency"])
    pf = pf.sort_values(sort_col)
 
    cols = [c for c in ["CustomerID", "Subscription", "Merchant", "Frequency",
                         "Next_Debit_Date", "Predicted_Amount", "Prediction_Method"] if c in pf.columns]
    st.dataframe(
        pf[cols].rename(columns={"Next_Debit_Date": "Next Debit", "Predicted_Amount": "Amount (Rs)"}),
        width='stretch', hide_index=True
    )
    st.download_button("⬇️ Export CSV", data=pf[cols].to_csv(index=False),
                       file_name="upcoming_debits.csv", mime="text/csv")
 
    # Chart 4 — Debit Timeline (BRD required)
    st.subheader("Debit Timeline")
    if "Next_Debit_Date" in pf.columns and "Predicted_Amount" in pf.columns:
        tl  = pf.sort_values("Next_Debit_Date")
        fig = px.line(tl, x="Next_Debit_Date", y="Predicted_Amount",
                      color="Merchant" if "Merchant" in tl.columns else None,
                      markers=True,
                      labels={"Next_Debit_Date": "Date", "Predicted_Amount": "Amount (Rs)"},
                      title="Upcoming Debit Timeline (next 30 days)")
        st.plotly_chart(fig, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK ALERTS
# - Counts per risk level shown after filter
# - Pie chart (or gauge for single customer) shown ABOVE the cards
# - Risk Score Distribution histogram REMOVED
# ══════════════════════════════════════════════════════════════════════════════
def tab_risk(data, sel_cust, sel_merch):
    risk = data.get("risk", pd.DataFrame())
    pred = data.get("pred", pd.DataFrame())
    if risk.empty:
        st.info("Run the pipeline to generate risk scores."); return
 
    rf = risk.copy()
    if sel_cust != "All Customers" and "CustomerID" in rf.columns:
        rf = rf[rf["CustomerID"] == sel_cust]
    if sel_merch != "All Merchants" and not pred.empty and "Merchant" in pred.columns:
        rf = rf[rf["CustomerID"].isin(
            pred[pred["Merchant"] == sel_merch]["CustomerID"].unique()
        )]
 
    lvl_filter = st.multiselect(
        "Filter by Risk Level", ["High", "Medium", "Low"], default=["High", "Medium"]
    )
    if lvl_filter:
        rf = rf[rf["Risk_Level"].isin(lvl_filter)]
    rf = rf.sort_values("Risk_Score", ascending=False)
 
    if rf.empty:
        st.info("No risk alerts for the selected filters."); return
 
    # Count per risk level after applying all filters
    counts = rf["Risk_Level"].value_counts()
    ck1, ck2, ck3 = st.columns(3)
    ck1.metric("🔴 High",   f"{counts.get('High',   0):,}")
    ck2.metric("🟡 Medium", f"{counts.get('Medium', 0):,}")
    ck3.metric("🟢 Low",    f"{counts.get('Low',    0):,}")
 
    st.divider()
 
    # Chart: gauge for single customer, pie for all/merchant
    is_single_customer = (sel_cust != "All Customers")
 
    if is_single_customer and not rf.empty:
        row   = rf.iloc[0]
        score = float(row.get("Risk_Score", 0))
        level = str(row.get("Risk_Level", "Low"))
        cid   = str(row.get("CustomerID", ""))
        st.subheader(f"Risk Score — {cid}")
        st.plotly_chart(render_gauge(score, level, title=f"{cid} | {level} Risk"),
                        use_container_width=True)
    else:
        # Pie chart — Chart 3 (BRD required)
        st.subheader("Risk Distribution")
        pie = rf["Risk_Level"].value_counts().reset_index()
        pie.columns = ["Risk_Level", "Count"]
        fig = px.pie(pie, names="Risk_Level", values="Count",
                     color="Risk_Level", color_discrete_map=RISK_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
 
    st.divider()
 
    # Risk alert cards
    for _, row in rf.head(20).iterrows():
        lvl    = str(row.get("Risk_Level", "Low"))
        score  = float(row.get("Risk_Score", 0))
        reason = str(row.get("Risk_Reason", "-"))
        cid    = str(row.get("CustomerID", "-"))
        bal    = float(row.get("Current_Balance", 0))
        upco   = float(row.get("Upcoming_Total_Debit", 0))
        subs   = int(row.get("Subscription_Count", 0))
        fail_r = float(row.get("Failed_Debit_Rate", 0))
        icon   = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(lvl, "⚪")
 
        with st.container(border=True):
            a, b = st.columns([4, 1])
            with a:
                st.markdown(f"**{icon} {cid}** — `{lvl} Risk`")
                st.caption(f"📋 Reason: {reason}")
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**Balance:** Rs{bal:,.0f}")
                c2.markdown(f"**Upcoming:** Rs{upco:,.0f}")
                c3.markdown(f"**Subscriptions:** {subs}")
                c4.markdown(f"**Fail Rate:** {fail_r*100:.1f}%")
            with b:
                st.metric("Score", f"{score:.4f}")
 
    st.divider()
    st.download_button(
        "⬇️ Export Risk CSV",
        data=rf[[c for c in ["CustomerID", "Risk_Level", "Risk_Score", "Risk_Reason",
                              "Current_Balance", "Upcoming_Total_Debit",
                              "Subscription_Count", "Failed_Debit_Rate"] if c in rf.columns]
                ].to_csv(index=False),
        file_name="risk_report.csv", mime="text/csv"
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONTHLY SPEND
# ══════════════════════════════════════════════════════════════════════════════
def tab_spend(data, sel_cust, d_from, d_to):
    ins     = data.get("insights", pd.DataFrame())
    summary = data.get("summary",  pd.DataFrame())
 
    ins_f = ins.copy()
    if sel_cust != "All Customers" and not ins_f.empty and "CustomerID" in ins_f.columns:
        ins_f = ins_f[ins_f["CustomerID"] == sel_cust]
 
    if ins_f.empty or "Active_Subscriptions" not in ins_f.columns:
        st.info("Run the pipeline to generate insights."); return
 
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Subscriptions / Customer", f"{ins_f['Active_Subscriptions'].mean():.1f}")
    k2.metric("Avg Monthly Spend",            f"Rs{ins_f['Total_Monthly_Spend'].mean():,.2f}")
    k3.metric("Max Monthly Spend",            f"Rs{ins_f['Total_Monthly_Spend'].max():,.2f}")
 
    if sel_cust != "All Customers" and not ins_f.empty:
        row = ins_f.iloc[0]
        msg = row.get(
            "FR8_Message",
            f"You have {row.get('Active_Subscriptions', 0)} active subscription(s). "
            f"Total monthly spend: Rs{row.get('Total_Monthly_Spend', 0):,.2f}."
        )
        st.info(f"📌 **FR8 Insight — {sel_cust}:** {msg}")
    else:
        st.info(
            f"📌 **FR8 Insight:** Average {ins_f['Active_Subscriptions'].mean():.1f} "
            f"active subscriptions | Average monthly spend "
            f"Rs{ins_f['Total_Monthly_Spend'].mean():,.2f}"
        )
 
    st.divider()
 
    if sel_cust == "All Customers":
        cl, cr = st.columns(2)
        with cl:
            st.subheader("Monthly Spend Distribution")
            fig = px.histogram(ins_f, x="Total_Monthly_Spend", nbins=30,
                               color_discrete_sequence=["#2ecc71"],
                               labels={"Total_Monthly_Spend": "Monthly Spend (Rs)"})
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            st.subheader("Top 15 Customers by Spend")
            top15 = ins_f.sort_values("Total_Monthly_Spend", ascending=False).head(15)
            fig2  = px.bar(top15, x="CustomerID", y="Total_Monthly_Spend",
                           color="Active_Subscriptions",
                           labels={"Total_Monthly_Spend": "Monthly Spend (Rs)"})
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        # Chart 2 — Spend by Merchant pie (BRD required)
        cs = summary[summary["CustomerID"] == sel_cust].copy() \
             if not summary.empty else pd.DataFrame()
        if not cs.empty:
            cl, cr = st.columns(2)
            with cl:
                st.subheader(f"Spend by Merchant — {sel_cust}")
                fig = px.pie(cs, names="Merchant", values="Avg_Amount",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
            with cr:
                disp = cs[["Description", "Merchant", "Frequency", "Avg_Amount"]].rename(
                    columns={"Description": "Subscription", "Avg_Amount": "Monthly Spend (Rs)"})
                disp.insert(0, "Type", "Subscription")
                st.dataframe(disp, width='stretch', hide_index=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GENAI ALERTS (FR7)
# Alert Validation and Pipeline Run Summary expanders REMOVED
# ══════════════════════════════════════════════════════════════════════════════
def tab_genai(data, sel_cust):
    risk        = data.get("risk", pd.DataFrame())
    alerts_path = os.path.join(RPT, "alerts.txt")
 
    if not os.path.exists(alerts_path) or os.path.getsize(alerts_path) == 0:
        st.warning("No alerts found.")
        st.info("Run `python run_pipeline.py` to generate alerts.")
        return
 
    with open(alerts_path, encoding="utf-8") as f:
        raw_txt = f.read()
 
    blocks, buf = [], []
    for line in raw_txt.splitlines():
        if line.startswith("=" * 28) and buf:
            txt = "\n".join(buf).strip()
            if len(txt) > 60: blocks.append(txt)
            buf = [line]
        else:
            buf.append(line)
    if buf:
        txt = "\n".join(buf).strip()
        if len(txt) > 60: blocks.append(txt)
 
    def get_cid(blk):
        for ln in blk.splitlines():
            m = re.search(r'CUSTOMER\s*[:\|]\s*(CUST\S+)', ln, re.IGNORECASE)
            if m: return m.group(1).strip()
        return None
 
    cid_blocks = [(b, get_cid(b)) for b in blocks if get_cid(b)]
    filtered   = [(b, c) for b, c in cid_blocks
                  if sel_cust == "All Customers" or c == sel_cust]
    if not filtered:
        st.warning(f"No alerts for {sel_cust}. Showing all.")
        filtered = cid_blocks
    if not filtered:
        st.warning("No alert blocks found. Re-run `python run_pipeline.py`.")
        return
 
    st.markdown(f"**{len(filtered)} alert(s) available**")
    opts   = [f"{c}  |  Alert {i+1}" for i, (_, c) in enumerate(filtered)]
    chosen = st.selectbox("Select Alert", opts)
    idx    = opts.index(chosen)
    blk, blk_cid = filtered[idx]
 
    cl, cr = st.columns([2, 1])
    with cl:
        st.code(blk, language=None)
    with cr:
        st.markdown(f"**👤 Customer:** `{blk_cid}`")
        if not risk.empty and "CustomerID" in risk.columns:
            rr = risk[risk["CustomerID"] == blk_cid]
            if not rr.empty:
                r    = rr.iloc[0]
                lvl  = str(r.get("Risk_Level", "-"))
                icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(lvl, "⚪")
                st.markdown(f"**{icon} Risk Level:** {lvl}")
                st.markdown(f"**Score:** `{r.get('Risk_Score', '-')}`")
                st.markdown(f"**Balance:** Rs{float(r.get('Current_Balance', 0)):,.0f}")
                st.markdown(f"**Upcoming:** Rs{float(r.get('Upcoming_Total_Debit', 0)):,.0f}")
                st.markdown(f"**Reason:** {r.get('Risk_Reason', '-')}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ADD NEW CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
def tab_add_customer():
    st.markdown(
        "Enter a new customer's transaction details. "
        "Runs **FR3 → FR4 → FR5 → FR6 → FR7** instantly."
    )
 
    with st.form("new_cust_form"):
        st.markdown("#### Customer Details")
        ca, cb = st.columns(2)
        with ca:
            nc_id   = st.text_input("CustomerID *",             placeholder="CUST999999")
            nc_desc = st.text_input("Transaction Description *", placeholder="NETFLIX MONTHLY SUBSCRIPTION")
            nc_mer  = st.text_input("Merchant Name *",           placeholder="Netflix")
        with cb:
            nc_amt  = st.number_input("Amount (Rs) *",  min_value=0.0, value=0.0, step=1.0)
            nc_bal  = st.number_input("Balance (Rs) *", min_value=0.0, value=0.0, step=100.0)
            # nc_fail = st.slider("Historical Failure Rate", 0.0, 0.3, 0.0, 0.01,
                                # help="Fraction of past debits that failed (0 = none)")
 
        st.markdown("#### Past Transaction Dates *(min 3 required — FR4)*")
        d1c, d2c, d3c, d4c = st.columns(4)
        nd1 = d1c.date_input("Date 1 *",          value=date(2025, 1, 15))
        nd2 = d2c.date_input("Date 2 *",          value=date(2025, 2, 15))
        nd3 = d3c.date_input("Date 3 *",          value=date(2025, 3, 15))
        nd4 = d4c.date_input("Date 4 (optional)", value=date(2025, 4, 15))
        submit  = st.form_submit_button("🚀 Analyse & Predict", type="primary",
                                    width='stretch')
 
    if submit:
        past = sorted(list({str(nd1), str(nd2), str(nd3), str(nd4)}))
        errs = validate_inputs(nc_id, nc_desc, nc_mer, nc_amt, nc_bal, past)
        if errs:
            for e in errs: st.error(f"❌ {e}")
            return
 
        with st.spinner("Running FR3 → FR4 → FR5 → FR6 → FR7..."):
            is_sub, conf             = nlp_classify(nc_desc)
            nxt_dt, nxt_amt, freq   = predict_next(past, nc_amt)
            r_score, r_level, r_rsn = compute_risk(float(nc_bal), float(nxt_amt),
                                                    fail_rate=0.0, sub_count=1)
            alert_txt               = build_alert(nc_id, nc_mer, nxt_dt, nxt_amt,
                                                   float(nc_bal), r_score, r_level, r_rsn)
        st.success("✅ Analysis complete!")
        st.divider()
 
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NLP Detection",    "Subscription ✅" if is_sub else "Non-Sub ❌",
                  f"{conf*100:.0f}% confidence")
        m2.metric("Next Debit Date",  str(nxt_dt))
        m3.metric("Predicted Amount", f"Rs{nxt_amt:,.2f}")
        icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(r_level, "⚪")
        m4.metric(f"{icon} Risk Level", r_level, f"Score: {r_score}")
 
        cl, cr = st.columns(2)
        with cl:
            st.markdown("#### FR3 – FR5: Detection & Prediction")
            st.dataframe(pd.DataFrame({
                "Field": ["CustomerID", "Description", "Merchant", "NLP Result",
                          "Confidence", "Frequency", "Next Debit Date", "Predicted Amount"],
                "Value": [nc_id, nc_desc, nc_mer,
                          "Subscription ✅" if is_sub else "Non-Subscription ❌",
                          f"{conf*100:.0f}%", freq, str(nxt_dt), f"Rs{nxt_amt:,.2f}"],
            }), width='stretch', hide_index=True)
        with cr:
            st.markdown("#### FR6: Risk Assessment")
            days_l = (pd.Timestamp(str(nxt_dt)).date() - date.today()).days
            st.dataframe(pd.DataFrame({
                "Field": ["Risk Score", "Risk Level", "Reason", "Balance",
                        "Upcoming Debit"],
                "Value": [str(r_score), r_level, r_rsn, f"Rs{nc_bal:,.2f}",
                        f"Rs{nxt_amt:,.2f}",
                        # f"{days_l}d" if days_l >= 0 else f"Overdue {abs(days_l)}d"
                        ],
            }), hide_index=True, use_container_width=True)
 
        st.markdown("#### FR7: GenAI Alert")
        st.code(alert_txt, language=None)
 
        st.markdown("#### Transaction Timeline")
        dts = [pd.Timestamp(str(d)) for d in past]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dts, y=[float(nc_amt)] * len(dts),
            mode="lines+markers", name="Historical",
            line=dict(color="#3498db", width=2), marker=dict(size=10)))
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(str(nxt_dt))], y=[nxt_amt],
            mode="markers+text", name="Predicted",
            marker=dict(size=14, color="#e74c3c", symbol="star"),
            text=[f"Rs{nxt_amt:.0f} (predicted)"], textposition="top center"))
        fig.update_layout(xaxis_title="Date", yaxis_title="Amount (Rs)",
                          title=f"Transaction Timeline — {nc_desc}")
        st.plotly_chart(fig, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def about_page():
    st.markdown("## ℹ️ About SubIntel")
    st.divider()
    with st.container(border=True):
        st.markdown("### 🏦 System Overview")
        st.markdown(
            "The **AI Subscription & Auto-Debit Intelligence System** helps banks proactively "
            "identify subscription transactions, predict upcoming debits, assess failure risk, "
            "and generate smart alerts — powered by NLP and machine learning."
        )
    with st.container(border=True):
        st.markdown("### ⚙️ Technology Stack")
        st.markdown("""
| FR | Component | Technology |
|---|---|---|
| FR3 | NLP Detection | spaCy + TF-IDF + Logistic Regression |
| FR4 | Pattern Detection | Date gap analysis (median gap) |
| FR5 | Debit Prediction | ARIMA(1,0,0) + Linear Regression blend |
| FR6 | Risk Scoring | Gradient Boosting Classifier |
| FR7 | GenAI Alerts | Microsoft Phi-2 (rule-based fallback) |
| FR8 | Insights | Customer monthly spend summary |
| FR9 | Dashboard | Streamlit + Plotly |
| FR1 | Data | Faker synthetic generator — 134K+ transactions |
""")
    with st.container(border=True):
        st.markdown("### 📋 BRD Compliance (FR1–FR9)")
        for fr, desc in [
            ("FR1", "Synthetic dataset — 134K+ transactions (Faker)"),
            ("FR2", "Data cleaning — null handling, text normalisation"),
            ("FR3", "NLP subscription detection — >90% accuracy (spaCy + TF-IDF + LR)"),
            ("FR4", "Recurring pattern detection — min 3 occurrences, gap-based frequency"),
            ("FR5", "Next debit prediction — ARIMA(1,0,0) + Linear Regression"),
            ("FR6", "Risk scoring — Gradient Boosting, >85% accuracy, explainable reasons"),
            ("FR7", "GenAI alerts — Phi-2 with structured rule-based fallback"),
            ("FR8", "Customer insights — monthly spend, active subscriptions"),
            ("FR9", "Interactive Streamlit dashboard with Plotly charts"),
        ]:
            st.markdown(f"✅ **{fr}:** {desc}")
    with st.container(border=True):
        st.markdown("### 👥 Team 7")
        st.markdown("**Mansi & Samyak** — AI Subscription & Auto-Debit Intelligence System")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def home_page(data):
    st.markdown("## 🏦 SubIntel — AI Subscription Intelligence Dashboard")
    st.divider()
 
    sel_cust, sel_merch, d_from, d_to = render_filters(data)
    st.divider()
    render_kpis(data, sel_cust, d_from, d_to)
    st.divider()
 
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Active Subscriptions",
        "Upcoming Debits",
        "Risk Alerts",
        "Monthly Spend",
        "Alerts",
        "Add New Customer",
    ])
    with tab1: tab_subscriptions(data, sel_cust, sel_merch, d_from, d_to)
    with tab2: tab_upcoming(data, sel_cust, sel_merch, d_from, d_to)
    with tab3: tab_risk(data, sel_cust, sel_merch)
    with tab4: tab_spend(data, sel_cust, d_from, d_to)
    with tab5: tab_genai(data, sel_cust)
    with tab6: tab_add_customer()
 
    st.divider()
    st.caption("© 2025 SubIntel — Team 7 (Mansi & Samyak) | AI Subscription & Auto-Debit Intelligence System")
 
# def home_page(data):
#     st.markdown("## 🏦 SubIntel — AI Subscription Intelligence Dashboard")
#     st.divider()
 
#     sel_cust, sel_merch, d_from, d_to = render_filters(data)
#     st.divider()
#     render_kpis(data, sel_cust, d_from, d_to)
#     st.divider()
 
#     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#         "🔄 Active Subscriptions",
#         "📅 Upcoming Debits",
#         "⚠️ Risk Alerts",
#         "📈 Monthly Spend",
#         "🤖 GenAI Alerts",
#         "➕ Add New Customer",
#     ])
#     with tab1: tab_subscriptions(data, sel_cust, sel_merch, d_from, d_to)
#     with tab2: tab_upcoming(data, sel_cust, sel_merch, d_from, d_to)
#     with tab3: tab_risk(data, sel_cust, sel_merch)
#     with tab4: tab_spend(data, sel_cust, d_from, d_to)
#     with tab5: tab_genai(data, sel_cust)
#     with tab6: tab_add_customer()
 
#     st.divider()
#     st.caption("© 2025 SubIntel — Team 7 (Mansi & Samyak) | AI Subscription & Auto-Debit Intelligence System")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# STATIC MATPLOTLIB REPORT  (called by run_pipeline.py for PNG report)
# ══════════════════════════════════════════════════════════════════════════════
def generate_static_dashboard(df, pred_df, risk_df, summary_df,
                               insights_df, output_path, nlp_metrics=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
 
    P = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71",
         "blue": "#2980b9", "dark": "#2c3e50"}
    fig = plt.figure(figsize=(24, 18), facecolor="white")
    fig.suptitle("AI Subscription & Auto-Debit Intelligence System\n"
                 "Team 7 - Mansi & Samyak | spaCy + ARIMA + Gradient Boosting + Phi-2",
                 fontsize=15, fontweight="bold", color=P["dark"], y=0.99)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.58, wspace=0.42)
 
    sub_pred = df["NLP_Sub_Pred"].sum() if "NLP_Sub_Pred" in df.columns \
               else df.get("SubscriptionFlag", pd.Series([0])).sum()
    high_cnt = (risk_df["Risk_Level"] == "High").sum() if not risk_df.empty else 0
 
    for col, (title, val, sub, clr) in enumerate([
        ("Total Transactions",  f"{len(df):,}",            "",           P["blue"]),
        ("Subscriptions",       f"{int(sub_pred):,}",      f"{sub_pred/len(df)*100:.1f}%", "#8e44ad"),
        ("Confirmed Recurring", f"{int(df['Is_Recurring'].sum()):,}", ">=3 occ", "#16a085"),
        ("High Risk",           f"{int(high_cnt):,}",      f"of {len(risk_df):,}", P["High"]),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.05, 0.05), 0.90, 0.90, boxstyle="round,pad=0.02",
                                    lw=2, edgecolor=clr, facecolor=clr + "25"))
        ax.text(0.5, 0.72, title, ha="center", fontsize=8.5, color=P["dark"], fontweight="bold")
        ax.text(0.5, 0.44, val,   ha="center", fontsize=18,  color=clr,       fontweight="bold")
        if sub: ax.text(0.5, 0.20, sub, ha="center", fontsize=7, color="grey")
 
    ax1 = fig.add_subplot(gs[1, 0:2])
    df["Date"] = pd.to_datetime(df["Date"])
    m = df.groupby(df["Date"].dt.to_period("M")).size().reset_index()
    m.columns = ["P", "C"]; m["P"] = m["P"].astype(str)
    ax1.fill_between(range(len(m)), m["C"], alpha=0.3, color=P["blue"])
    ax1.plot(range(len(m)), m["C"], color=P["blue"], lw=2)
    ax1.set_xticks(range(0, len(m), 3))
    ax1.set_xticklabels(m["P"].iloc[::3], rotation=45, fontsize=7)
    ax1.set_title("Transaction Volume", fontweight="bold", fontsize=10)
    ax1.set_ylabel("Count"); ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
 
    ax2 = fig.add_subplot(gs[1, 2])
    if not risk_df.empty and "Risk_Level" in risk_df.columns:
        rc = risk_df["Risk_Level"].value_counts()
        ax2.pie(rc.values, labels=rc.index,
                colors=[P.get(l, "#aaa") for l in rc.index],
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax2.set_title("Risk Distribution", fontweight="bold", fontsize=10)
 
    ax3 = fig.add_subplot(gs[1, 3])
    if nlp_metrics:
        keys = ["accuracy", "precision", "recall", "f1"]
        vals = [nlp_metrics.get(k, 0) for k in keys]
        bars = ax3.bar(["Acc", "Prec", "Rec", "F1"], vals,
                       color=[P["blue"] if v >= 0.90 else P["Medium"] for v in vals],
                       edgecolor="white")
        ax3.axhline(0.90, color=P["High"], ls="--", lw=1.2)
        ax3.set_ylim(0, 1.12)
        ax3.set_title("NLP Metrics (FR3)", fontweight="bold", fontsize=10)
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax3.spines[["top", "right"]].set_visible(False)
 
    ax4 = fig.add_subplot(gs[2, 0:2])
    if not summary_df.empty:
        ts = summary_df.groupby("Description")["Occurrences"].sum() \
                       .sort_values(ascending=False).head(10)
        ax4.barh([d[:32] for d in ts.index][::-1], ts.values[::-1],
                 color=P["blue"], edgecolor="white")
        ax4.set_title("Top 10 Subscriptions", fontweight="bold", fontsize=10)
        ax4.set_xlabel("Occurrences")
        ax4.spines[["top", "right"]].set_visible(False)
 
    ax5 = fig.add_subplot(gs[2, 2])
    if "Is_Recurring" in df.columns and "Inferred_Freq" in df.columns:
        fd = df[df["Is_Recurring"] == 1]["Inferred_Freq"].value_counts()
        if not fd.empty:
            ax5.bar(fd.index, fd.values,
                    color=["#16a085", "#8e44ad"][:len(fd)], edgecolor="white")
    ax5.set_title("Recurring Frequency (FR4)", fontweight="bold", fontsize=10)
    ax5.set_ylabel("Count")
    ax5.spines[["top", "right"]].set_visible(False)
 
    ax6 = fig.add_subplot(gs[2, 3])
    if not risk_df.empty and "Risk_Score" in risk_df.columns:
        ax6.hist(risk_df["Risk_Score"], bins=30, color=P["blue"], edgecolor="white", alpha=0.8)
        ax6.axvline(0.35, color=P["Medium"], ls="--", lw=1.5, label="Med")
        ax6.axvline(0.65, color=P["High"],   ls="--", lw=1.5, label="High")
        ax6.set_title("Risk Score Distribution (FR6)", fontweight="bold", fontsize=10)
        ax6.set_xlabel("Score"); ax6.set_ylabel("Accounts"); ax6.legend(fontsize=7)
        ax6.spines[["top", "right"]].set_visible(False)
 
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis("off")
    if not risk_df.empty:
        tr = risk_df.sort_values("Risk_Score", ascending=False).head(8)[
            ["CustomerID", "Current_Balance", "Upcoming_Total_Debit",
             "Failed_Debit_Rate", "Subscription_Count", "Risk_Score", "Risk_Level"]].copy()
        tr["Current_Balance"]      = tr["Current_Balance"].map("Rs{:,.0f}".format)
        tr["Upcoming_Total_Debit"] = tr["Upcoming_Total_Debit"].map("Rs{:,.0f}".format)
        tr["Failed_Debit_Rate"]    = tr["Failed_Debit_Rate"].map("{:.1%}".format)
        tr["Risk_Score"]           = tr["Risk_Score"].map("{:.4f}".format)
        tr["Subscription_Count"]   = tr["Subscription_Count"].astype(int)
        tbl = ax7.table(
            cellText=tr.values,
            colLabels=["CustomerID", "Balance", "Upcoming", "Fail%", "Subs", "Score", "Level"],
            cellLoc="center", loc="center", bbox=[0, 0.05, 1, 0.90]
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor(P["dark"])
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f5f5f5")
        ax7.set_title("Top High-Risk Accounts", fontweight="bold", fontsize=10, pad=8)
 
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[Dashboard] Saved -> {output_path}")
    plt.close()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_dashboard():
    st.set_page_config(
        page_title="SubIntel - AI Subscription Intelligence",
        page_icon="🏦",
        layout="wide",
    )
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
 
    if not st.session_state["logged_in"]:
        login_page()
        return
 
    render_sidebar()
    data = load_data()
 
    if st.session_state.get("page") == "About":
        about_page()
    else:
        home_page(data)
 
 
if __name__ == "__main__":
    run_dashboard()
 