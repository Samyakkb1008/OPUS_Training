"""
FR4 - Recurring Pattern Detection
Detects weekly/monthly recurring subscriptions.
Requires at least 3 occurrences with consistent gaps.
"""

import pandas as pd
import numpy as np

MIN_OCCURRENCES = 3
WEEKLY_GAP  = (5, 9)    # days
MONTHLY_GAP = (25, 35)  # days


def get_frequency(gaps):
    if not gaps:
        return "Irregular"
    med = np.median(gaps)
    if WEEKLY_GAP[0]  <= med <= WEEKLY_GAP[1]:  return "Weekly"
    if MONTHLY_GAP[0] <= med <= MONTHLY_GAP[1]: return "Monthly"
    return "Irregular"


def detect_recurring_patterns(df):
    print("\n[FR4] Detecting recurring patterns...")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Is_Recurring"]     = 0
    df["Inferred_Freq"]    = "None"
    df["Occurrence_Count"] = 0

    pred_col = "NLP_Sub_Pred" if "NLP_Sub_Pred" in df.columns else "SubscriptionFlag"
    subs = df[(df[pred_col] == 1) & (df["TransactionType"] == "DEBIT")]

    summary_rows = []

    for (cust_id, desc), group in subs.groupby(["CustomerID", "Description_Clean"]):
        group = group.sort_values("Date")
        dates = group["Date"].tolist()

        if len(dates) < MIN_OCCURRENCES:
            continue

        gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        freq = get_frequency(gaps)

        if freq == "Irregular":
            continue

        df.loc[group.index, "Is_Recurring"]     = 1
        df.loc[group.index, "Inferred_Freq"]    = freq
        df.loc[group.index, "Occurrence_Count"] = len(dates)

        summary_rows.append({
            "CustomerID":      cust_id,
            "Description":     desc,
            "Merchant":        group["Merchant"].mode().iloc[0],
            "Frequency":       freq,
            "Occurrences":     len(dates),
            "First_Date":      dates[0].date(),
            "Last_Date":       dates[-1].date(),
            "Avg_Amount":      round(group["Amount"].mean(), 2),
            "Std_Amount":      round(group["Amount"].std(), 2),
            "Median_Gap_Days": round(np.median(gaps), 1),
            "Failed_Count":    int((group["Status"] == "FAILED").sum()),
        })

    summary_df = pd.DataFrame(summary_rows)

    print(f"  Confirmed recurring groups : {len(summary_df)}")
    print(f"  Rows flagged               : {df['Is_Recurring'].sum()}")

    # FR8 preview — per-customer monthly subscription summary
    insights_df = pd.DataFrame()
    if not summary_df.empty:
        monthly = summary_df[summary_df["Frequency"] == "Monthly"]
        insights_df = (
            monthly.groupby("CustomerID")
            .agg(Active_Subscriptions=("Description", "count"),
                 Total_Monthly_Spend =("Avg_Amount",  "sum"),
                 Merchants           =("Merchant",    lambda x: ", ".join(x.unique()[:3])))
            .reset_index()
        )

    return df, summary_df, insights_df


#################################################################################################

# """
# FR4 - Recurring Pattern Detection
# Groups transactions by customer + description, checks if they repeat at
# regular intervals (weekly or monthly). Requires at least 3 occurrences.
# """

# import pandas as pd
# import numpy as np


# MIN_OCCURRENCES = 3        # Need at least 3 to confirm recurring
# WEEKLY_GAP      = (5, 9)   # 5-9 days = weekly
# MONTHLY_GAP     = (25, 35) # 25-35 days = monthly


# def _get_frequency(gaps):
#     """Determine Weekly / Monthly / Irregular based on median gap between transactions."""
#     if not gaps:
#         return "Irregular"
#     median = np.median(gaps)
#     if WEEKLY_GAP[0] <= median <= WEEKLY_GAP[1]:
#         return "Weekly"
#     if MONTHLY_GAP[0] <= median <= MONTHLY_GAP[1]:
#         return "Monthly"
#     return "Irregular"


# def detect_recurring_patterns(df):
#     """
#     Find all confirmed recurring subscriptions.

#     Logic:
#     - Look at each customer's subscription transactions grouped by description
#     - Calculate gaps between consecutive transactions
#     - If >= 3 occurrences and gaps are consistently weekly or monthly → mark as recurring

#     Returns:
#         df           : original df with Is_Recurring, Inferred_Freq, Occurrence_Count added
#         summary_df   : one row per confirmed recurring subscription
#         insights_df  : per-customer subscription summary (used by FR8)
#     """
#     print(f"\n{'='*60}")
#     print("  FR4 - RECURRING PATTERN DETECTION")
#     print(f"{'='*60}")

#     df = df.copy()
#     df["Date"] = pd.to_datetime(df["Date"])

#     # New columns to be filled
#     df["Is_Recurring"]     = 0
#     df["Inferred_Freq"]    = "None"
#     df["Occurrence_Count"] = 0

#     # Only look at NLP-predicted subscription debits
#     pred_col = "NLP_Sub_Pred" if "NLP_Sub_Pred" in df.columns else "SubscriptionFlag"
#     subs = df[(df[pred_col] == 1) & (df["TransactionType"] == "DEBIT")]

#     summary_rows     = []
#     total_groups     = 0
#     confirmed_groups = 0

#     for (cust_id, desc), group in subs.groupby(["CustomerID", "Description_Clean"]):
#         total_groups += 1
#         group  = group.sort_values("Date")
#         dates  = group["Date"].tolist()

#         # Skip if not enough occurrences
#         if len(dates) < MIN_OCCURRENCES:
#             continue

#         gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
#         freq = _get_frequency(gaps)

#         # Skip if gaps are irregular
#         if freq == "Irregular":
#             continue

#         confirmed_groups += 1

#         # Mark these rows in the main dataframe
#         df.loc[group.index, "Is_Recurring"]     = 1
#         df.loc[group.index, "Inferred_Freq"]    = freq
#         df.loc[group.index, "Occurrence_Count"] = len(dates)

#         summary_rows.append({
#             "CustomerID":      cust_id,
#             "Description":     desc,
#             "Merchant":        group["Merchant"].mode().iloc[0] if not group.empty else "Unknown",
#             "Frequency":       freq,
#             "Occurrences":     len(dates),
#             "First_Date":      dates[0].date(),
#             "Last_Date":       dates[-1].date(),
#             "Avg_Amount":      round(group["Amount"].mean(), 2),
#             "Std_Amount":      round(group["Amount"].std(), 2),
#             "Median_Gap_Days": round(np.median(gaps), 1),
#             "Failed_Count":    int((group["Status"] == "FAILED").sum()),
#         })

#     summary_df = pd.DataFrame(summary_rows)

#     print(f"\n  Groups checked        : {total_groups:,}")
#     print(f"  Confirmed recurring   : {confirmed_groups:,}")
#     print(f"  Rows flagged          : {df['Is_Recurring'].sum():,}")

#     if not summary_df.empty:
#         print(f"\n  Frequency breakdown:")
#         for freq, count in summary_df["Frequency"].value_counts().items():
#             print(f"    {freq:<12}: {count:,}")

#     # BRD §8 Scenario 2: detect multiple subscriptions on the same date
#     recurring_debits = df[(df["Is_Recurring"] == 1) & (df["TransactionType"] == "DEBIT")]
#     same_date = (recurring_debits.groupby(["CustomerID", "Date"])
#                  .size().reset_index(name="count"))
#     multi = same_date[same_date["count"] > 1]
#     print(f"\n  [Scenario 2] Customers with multiple subs on same date: {multi['CustomerID'].nunique():,}")

#     print(f"\n  Sample confirmed subscriptions:")
#     if not summary_df.empty:
#         print(summary_df[["CustomerID","Description","Frequency","Occurrences",
#                            "Avg_Amount","Median_Gap_Days"]].head(6).to_string(index=False))

#     # FR8 preview: per-customer subscription count and spend
#     insights_df = pd.DataFrame()
#     if not summary_df.empty:
#         monthly = summary_df[summary_df["Frequency"] == "Monthly"]
#         insights_df = (
#             monthly.groupby("CustomerID")
#             .agg(Active_Subscriptions=("Description", "count"),
#                  Total_Monthly_Spend =("Avg_Amount",  "sum"),
#                  Merchants           =("Merchant",    lambda x: ", ".join(x.unique()[:3])))
#             .reset_index()
#         )
#         insights_df["Total_Monthly_Spend"] = insights_df["Total_Monthly_Spend"].round(2)

#         print(f"\n  [FR8 Preview] Avg subs/customer : {insights_df['Active_Subscriptions'].mean():.1f}")
#         print(f"  [FR8 Preview] Avg monthly spend : ₹{insights_df['Total_Monthly_Spend'].mean():,.2f}")

#     print(f"{'='*60}\n")
#     return df, summary_df, insights_df


# if __name__ == "__main__":
#     df = pd.read_csv("../data/transactions_nlp.csv")
#     df, summary, insights = detect_recurring_patterns(df)
#     print(summary.head())
