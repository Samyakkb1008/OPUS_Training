"""
FR2 - Data Cleaning & Preprocessing
Fixes nulls, normalises text descriptions, removes duplicates.
Shows before/after null counts to prove cleaning happened.
"""

import pandas as pd
import numpy as np
import re


# Abbreviation expansions for text normalisation
ABBREVIATIONS = {
    r'\bMTHLY\b': 'MONTHLY', r'\bMO\b':    'MONTHLY',
    r'\bWKLY\b':  'WEEKLY',  r'\bSUBS?\b': 'SUBSCRIPTION',
    r'\bPYMT\b':  'PAYMENT', r'\bPMT\b':   'PAYMENT',
    r'\bRCRNG\b': 'RECURRING',r'\bREC\b':  'RECURRING',
    r'\bDR\b':    'DEBIT',   r'\bCR\b':    'CREDIT',
    r'\bTRFR\b':  'TRANSFER',r'\bWDL\b':   'WITHDRAWAL',
}


def _clean_text(text):
    """Normalise a transaction description: uppercase, expand abbreviations, strip noise."""
    if pd.isna(text) or not isinstance(text, str):
        return "UNKNOWN TRANSACTION"
    t = text.upper().strip()
    for pattern, replacement in ABBREVIATIONS.items():
        t = re.sub(pattern, replacement, t)
    t = re.sub(r'[^A-Z0-9\s\.\-/]', ' ', t)   # remove special symbols
    t = re.sub(r'\s{2,}', ' ', t).strip()
    return t


def clean_data(df):
    """
    Clean the raw transaction dataset.
    Steps: fill nulls → normalise text → fix dtypes → remove duplicates.
    Returns cleaned DataFrame with a new Description_Clean column.
    """
    print(f"\n{'='*60}")
    print("  FR2 - DATA CLEANING & PREPROCESSING")
    print(f"{'='*60}")

    df = df.copy()
    total = len(df)
    print(f"\n  Input rows : {total:,}")

    # Show null counts BEFORE cleaning
    critical = ["Description", "Amount", "Balance"]
    print("\n  [Before] Null counts:")
    for col in critical:
        n = df[col].isna().sum()
        print(f"    {col:<20} {n:>6,}  ({n/total*100:.2f}%)")

    # Step 1: Fill Amount nulls using median of same merchant
    null_amt = df["Amount"].isna().sum()
    df["Amount"] = df["Amount"].fillna(df.groupby("Merchant")["Amount"].transform("median"))
    df["Amount"] = df["Amount"].fillna(df["Amount"].median())
    print(f"\n  [Step 1] Amount nulls filled   : {null_amt:,}  (merchant median)")

    # Step 2: Fill Balance nulls using forward-fill within each customer account
    null_bal = df["Balance"].isna().sum()
    df = df.sort_values(["CustomerID", "Date"])
    df["Balance"] = df.groupby("CustomerID")["Balance"].transform(lambda s: s.ffill().bfill())
    df["Balance"] = df["Balance"].fillna(df["Balance"].median())
    print(f"  [Step 2] Balance nulls filled  : {null_bal:,}  (forward-fill per customer)")

    # Step 3: Fill Description nulls with placeholder
    null_desc = df["Description"].isna().sum()
    df["Description"] = df["Description"].fillna("UNKNOWN TRANSACTION")
    print(f"  [Step 3] Description nulls     : {null_desc:,}  → 'UNKNOWN TRANSACTION'")

    # Step 4: Create cleaned description column
    df["Description_Clean"] = df["Description"].apply(_clean_text)
    print(f"  [Step 4] Description_Clean column created")

    # Step 5: Fix data types
    df["Date"]    = pd.to_datetime(df["Date"])
    df["Amount"]  = df["Amount"].astype(float).round(2)
    df["Balance"] = df["Balance"].astype(float).round(2)
    print(f"  [Step 5] Dtypes fixed: Date → datetime, Amount/Balance → float")

    # Step 6: Remove duplicate transactions
    dups = df.duplicated(subset=["TransactionID"]).sum()
    df   = df.drop_duplicates(subset=["TransactionID"]).reset_index(drop=True)
    print(f"  [Step 6] Duplicates removed    : {dups:,}")

    # Show null counts AFTER cleaning
    remaining = df[critical].isna().sum().sum()
    print(f"\n  [After]  Output rows : {len(df):,}")
    print(f"  [After]  Nulls left  : {remaining}  {'✅' if remaining == 0 else '⚠️'}")

    # Show a few normalisation examples
    print("\n  [Sample] Description normalisation:")
    sample = df[["Description","Description_Clean"]].drop_duplicates().head(5)
    for _, row in sample.iterrows():
        print(f"    '{row['Description']}'  →  '{row['Description_Clean']}'")

    print(f"{'='*60}\n")
    return df


if __name__ == "__main__":
    raw     = pd.read_csv("../data/transactions_raw.csv")
    cleaned = clean_data(raw)
    cleaned.to_csv("../data/transactions_cleaned.csv", index=False)
