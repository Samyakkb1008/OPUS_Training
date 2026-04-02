"""
FR8 - Insights Generation
Per-customer subscription summary.
BRD output: "You have 5 active subscriptions. Total monthly spend: ₹2,500."
"""

import pandas as pd


def generate_insights(df, summary_df, risk_df, pred_df):
    print("\n[FR8] Generating insights...")

    if summary_df.empty:
        print("  No data — run FR4 first.")
        return pd.DataFrame()

    rows = []
    for cust_id, group in summary_df.groupby("CustomerID"):
        monthly = group[group["Frequency"] == "Monthly"]
        weekly  = group[group["Frequency"] == "Weekly"]

        # Weekly spend × 4 ≈ monthly equivalent
        total_spend = monthly["Avg_Amount"].sum() + weekly["Avg_Amount"].sum() * 4

        risk_row = risk_df[risk_df["CustomerID"] == cust_id]
        risk_lvl = risk_row["Risk_Level"].iloc[0]     if not risk_row.empty else "Unknown"
        balance  = risk_row["Current_Balance"].iloc[0] if not risk_row.empty else 0

        up_rows      = pred_df[pred_df["CustomerID"] == cust_id]
        upcoming_amt = up_rows["Predicted_Amount"].sum() if not up_rows.empty else 0
        next_due     = up_rows["Next_Debit_Date"].min()  if not up_rows.empty else None

        fr8_msg = (f"You have {len(group)} active subscription(s). "
                   f"Total monthly spend: ₹{total_spend:,.2f}. "
                   f"Next debit: {next_due}.")

        rows.append({
            "CustomerID":           cust_id,
            "Active_Subscriptions": len(group),
            "Monthly_Subscriptions":len(monthly),
            "Weekly_Subscriptions": len(weekly),
            "Total_Monthly_Spend":  round(total_spend, 2),
            "Upcoming_Total":       round(upcoming_amt, 2),
            "Next_Debit_Date":      next_due,
            "Current_Balance":      round(balance, 2),
            "Risk_Level":           risk_lvl,
            "Top_Merchants":        ", ".join(group["Merchant"].unique()[:5]),
            "FR8_Message":          fr8_msg,
        })

    insights_df = pd.DataFrame(rows)

    print(f"  Customers with subscriptions : {len(insights_df)}")
    if not insights_df.empty:
        print(f"  Avg subscriptions/customer   : {insights_df['Active_Subscriptions'].mean():.1f}")
        print(f"  Avg monthly spend            : ₹{insights_df['Total_Monthly_Spend'].mean():,.2f}")
        print("\n  Sample FR8 messages:")
        for _, row in insights_df.head(3).iterrows():
            print(f"    [{row['CustomerID']}] {row['FR8_Message']}")

    return insights_df


##############################################################################################


# """
# FR8 - Insights Generation
# Produces per-customer subscription summaries.
# BRD FR8 output: "You have 5 active subscriptions. Total monthly spend: ₹2,500."
# """

# import pandas as pd


# def generate_insights(df, summary_df, risk_df, pred_df):
#     """
#     Build one insight row per customer with:
#     - subscription counts (monthly + weekly)
#     - total monthly spend
#     - upcoming debit total and next due date
#     - risk level and balance
#     - FR8 message string (BRD format)

#     Returns a DataFrame with one row per customer.
#     """
#     print(f"\n{'='*60}")
#     print("  FR8 - INSIGHTS GENERATION")
#     print(f"{'='*60}")

#     if summary_df.empty:
#         print("  [FR8] No data available — run FR4 first.")
#         return pd.DataFrame()

#     rows = []
#     for cust_id, group in summary_df.groupby("CustomerID"):
#         monthly = group[group["Frequency"] == "Monthly"]
#         weekly  = group[group["Frequency"] == "Weekly"]

#         # Calculate total monthly spend (weekly spend × 4 = monthly equivalent)
#         total_spend = monthly["Avg_Amount"].sum() + weekly["Avg_Amount"].sum() * 4

#         # Get risk info for this customer
#         risk_row = risk_df[risk_df["CustomerID"] == cust_id]
#         risk_lvl = risk_row["Risk_Level"].iloc[0]    if not risk_row.empty else "Unknown"
#         balance  = risk_row["Current_Balance"].iloc[0] if not risk_row.empty else 0

#         # Get upcoming debit info
#         up_rows      = pred_df[pred_df["CustomerID"] == cust_id]
#         upcoming_amt = up_rows["Predicted_Amount"].sum() if not up_rows.empty else 0
#         next_due     = up_rows["Next_Debit_Date"].min()  if not up_rows.empty else None

#         # BRD FR8 message format
#         fr8_msg = (f"You have {len(group)} active subscription(s). "
#                    f"Total monthly spend: ₹{total_spend:,.2f}. "
#                    f"Next debit: {next_due}.")

#         rows.append({
#             "CustomerID":           cust_id,
#             "Active_Subscriptions": len(group),
#             "Monthly_Subscriptions":len(monthly),
#             "Weekly_Subscriptions": len(weekly),
#             "Total_Monthly_Spend":  round(total_spend, 2),
#             "Upcoming_Total":       round(upcoming_amt, 2),
#             "Next_Debit_Date":      next_due,
#             "Current_Balance":      round(balance, 2),
#             "Risk_Level":           risk_lvl,
#             "Top_Merchants":        ", ".join(group["Merchant"].unique()[:5]),
#             "FR8_Message":          fr8_msg,
#         })

#     insights_df = pd.DataFrame(rows)

#     # System-level summary
#     print(f"\n  Customers with subscriptions : {len(insights_df):,}")
#     if not insights_df.empty:
#         print(f"  Avg subscriptions/customer   : {insights_df['Active_Subscriptions'].mean():.1f}")
#         print(f"  Avg monthly spend            : ₹{insights_df['Total_Monthly_Spend'].mean():,.2f}")
#         print(f"  Max monthly spend            : ₹{insights_df['Total_Monthly_Spend'].max():,.2f}")
#         print(f"\n  Sample FR8 messages:")
#         for _, row in insights_df.head(4).iterrows():
#             print(f"    [{row['CustomerID']}] {row['FR8_Message']}")

#     print(f"{'='*60}\n")
#     return insights_df


# if __name__ == "__main__":
#     summary = pd.read_csv("../data/recurring_summary.csv")
#     risk    = pd.read_csv("../data/risk_scored.csv")
#     pred    = pd.read_csv("../data/predictions.csv")
#     df      = pd.read_csv("../data/transactions_patterns.csv")
#     ins     = generate_insights(df, summary, risk, pred)
#     print(ins[["CustomerID","Active_Subscriptions","Total_Monthly_Spend","FR8_Message"]].head())
