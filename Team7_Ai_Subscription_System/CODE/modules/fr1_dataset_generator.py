"""
FR1 - Synthetic Dataset Generator
Generates realistic banking transactions for 1200 customers.
Includes: salary credits, subscriptions, random transactions, failed debits, and nulls.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Try Faker for realistic Indian names, fall back to simple lists
try:
    from faker import Faker
    fake = Faker("en_IN")
    USE_FAKER = True
except ImportError:
    USE_FAKER = False

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Subscription catalogue ─────────────────────────────────────────────────────
SUBSCRIPTIONS = [
    ("NETFLIX.COM MONTHLY SUB",    "Netflix",        649.0,  "monthly"),
    ("SPOTIFY PREMIUM MONTHLY",    "Spotify",        119.0,  "monthly"),
    ("AMAZON PRIME SUBSCRIPTION",  "Amazon",         299.0,  "monthly"),
    ("YOUTUBE PREMIUM MONTHLY",    "YouTube",        189.0,  "monthly"),
    ("HOTSTAR PREMIUM",            "Disney+Hotstar", 1499.0, "monthly"),
    ("APPLE ICLOUD STORAGE",       "Apple",           75.0,  "monthly"),
    ("GOOGLE ONE STORAGE PLAN",    "Google",         130.0,  "monthly"),
    ("MICROSOFT 365 SUBSCRIPTION", "Microsoft",      499.0,  "monthly"),
    ("ZEE5 PREMIUM PLAN",          "Zee5",            99.0,  "monthly"),
    ("SWIGGY ONE MEMBERSHIP",      "Swiggy",         179.0,  "monthly"),
    ("LINKEDIN PREMIUM CAREER",    "LinkedIn",      2650.0,  "monthly"),
    ("DROPBOX PLUS PLAN",          "Dropbox",        899.0,  "monthly"),
    ("ADOBE CREATIVE CLOUD",       "Adobe",         1675.0,  "monthly"),
    ("ZOOM PRO MONTHLY",           "Zoom",          1300.0,  "monthly"),
    ("COURSERA PLUS MONTHLY",      "Coursera",      3999.0,  "monthly"),
    ("ELECTRICITY BILL AUTO PAY",  "BESCOM",         850.0,  "monthly"),
    ("GYM MEMBERSHIP MONTHLY",     "CultFit",        999.0,  "monthly"),
    ("NEWSPAPER DIGITAL WEEKLY",   "TimesOfIndia",    49.0,  "weekly"),
    ("GYM LOCKER WEEKLY",          "AnytimeFitness", 249.0,  "weekly"),
]

# Ambiguous descriptions — edge cases for NLP (BRD FR3)
AMBIGUOUS = [
    ("AUTO PAY 4521",          "AutoPay"),
    ("ACH DEBIT 00293",        "ACH"),
    ("NACH DEBIT MANDATE",     "NACH"),
    ("STANDING INSTRUCTION",   "SI"),
    ("RECURRING TRANSFER REF", "Bank"),
    ("PERIODIC PAYMENT 7734",  "AutoDebit"),
]

# Non-subscription transactions
NON_SUBS = [
    ("MONTHLY SALARY CREDIT",  "Employer",  "CREDIT"),
    ("ATM CASH WITHDRAWAL",    "ATM",       "DEBIT"),
    ("GROCERY PURCHASE",       "BigBasket", "DEBIT"),
    ("FUEL PETROL PUMP",       "HPCL",      "DEBIT"),
    ("RESTAURANT SWIGGY",      "Swiggy",    "DEBIT"),
    ("MOBILE RECHARGE",        "Jio",       "DEBIT"),
    ("ONLINE SHOPPING",        "Flipkart",  "DEBIT"),
    ("UPI TRANSFER",           "PhonePe",   "DEBIT"),
    ("NEFT TRANSFER OUTWARD",  "Bank",      "DEBIT"),
    ("MEDICAL PHARMACY",       "MedPlus",   "DEBIT"),
    ("INTEREST CREDIT",        "Bank",      "CREDIT"),
    ("TAX REFUND CREDIT",      "IncomeTax", "CREDIT"),
    ("INSURANCE PREMIUM LIC",  "LIC",       "DEBIT"),
    ("LOAN EMI SBI",           "SBI",       "DEBIT"),
    ("RENT PAYMENT NEFT",      "Landlord",  "DEBIT"),
]

FIRST_NAMES = ["Aarav","Ananya","Rohan","Priya","Vikram","Meera","Arjun","Sneha",
               "Karan","Pooja","Rahul","Deepa","Amit","Kavya","Suresh","Lakshmi"]
LAST_NAMES  = ["Sharma","Patel","Singh","Nair","Reddy","Gupta","Kumar","Verma",
               "Joshi","Shah","Das","Rao","Pillai","Mehta","Chopra","Iyer"]


def _random_name():
    if USE_FAKER:
        return fake.name()
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _add_noise(amount, pct=0.04):
    """Add small random variation to simulate slight price changes."""
    return round(amount * (1 + np.random.uniform(-pct, pct)), 2)


def _next_month(dt):
    """Advance date by one month, safely handling Feb/month-end edge cases."""
    m = dt.month + 1
    y = dt.year + (1 if m > 12 else 0)
    m = m if m <= 12 else 1
    d = min(dt.day, 28 if m == 2 else 30)
    return dt.replace(year=y, month=m, day=d)


def _make_row(cust_id, name, dt, desc, amount, balance, merchant, txn_type, is_sub, status, freq):
    """Build a single transaction dictionary."""
    return {
        "CustomerID":       cust_id,
        "CustomerName":     name,
        "TransactionID":    f"TXN{random.randint(10_000_000, 99_999_999)}",
        "Date":             dt.strftime("%Y-%m-%d"),
        "Description":      desc,
        "Amount":           amount,
        "Balance":          balance,
        "Merchant":         merchant,
        "TransactionType":  txn_type,
        "SubscriptionFlag": is_sub,
        "Status":           status,
        "Frequency":        freq,
    }


def _customer_transactions(cust_id, name, start, end, subs):
    """Generate all transactions for one customer."""
    rows    = []
    balance = round(np.random.uniform(8_000, 200_000), 2)
    salary  = round(np.random.uniform(25_000, 150_000), 2)

    # Monthly salary (keeps balance realistic)
    sal_dt = start.replace(day=1) + timedelta(days=random.randint(0, 3))
    while sal_dt <= end:
        balance += salary
        rows.append(_make_row(cust_id, name, sal_dt,
                              "MONTHLY SALARY CREDIT", salary, round(balance, 2),
                              "Employer", "CREDIT", 0, "SUCCESS", "monthly"))
        sal_dt = _next_month(sal_dt)

    # Recurring subscription debits
    for (desc, merchant, base_amt, freq) in subs:
        dt = start + timedelta(days=random.randint(0, 27))
        while dt <= end:
            failed = random.random() < 0.07     # 7% failure rate
            amt    = _add_noise(base_amt)
            if not failed:
                balance -= amt

            # Inject nulls at low rates to test FR2 cleaning
            rows.append(_make_row(
                cust_id, name, dt,
                None if random.random() < 0.008 else desc,
                None if random.random() < 0.022 else round(amt, 2),
                None if random.random() < 0.020 else round(balance, 2),
                merchant, "DEBIT", 1,
                "FAILED" if failed else "SUCCESS", freq
            ))
            dt = _next_month(dt) if freq == "monthly" else dt + timedelta(weeks=1)

    # Ambiguous edge-case transactions (AUTO PAY, ACH DEBIT, etc.)
    for (desc, merchant) in random.sample(AMBIGUOUS, k=random.randint(1, 3)):
        dt  = start + timedelta(days=random.randint(0, (end - start).days))
        amt = round(np.random.uniform(100, 3000), 2)
        balance -= amt
        rows.append(_make_row(cust_id, name, dt,
                              desc, round(amt, 2), round(balance, 2),
                              merchant, "DEBIT", 0, "SUCCESS", "monthly"))

    # Regular non-subscription transactions
    for _ in range(random.randint(15, 50)):
        dt = start + timedelta(days=random.randint(0, (end - start).days))
        (desc, merchant, txn_type) = random.choice(NON_SUBS)
        amt = round(np.random.lognormal(7.5, 1.2), 2)
        balance = balance - amt if txn_type == "DEBIT" else balance + amt

        rows.append(_make_row(
            cust_id, name, dt,
            None if random.random() < 0.006 else desc,
            None if random.random() < 0.025 else round(amt, 2),
            None if random.random() < 0.018 else round(balance, 2),
            merchant, txn_type, 0, "SUCCESS", "none"
        ))

    return rows


def generate_dataset(n_accounts=1200, output_path=None):
    """Generate the full synthetic banking transaction dataset."""
    print(f"\n{'='*60}")
    print("  FR1 - SYNTHETIC DATASET GENERATION")
    print(f"{'='*60}")

    start, end = datetime(2023, 1, 1), datetime(2024, 6, 30)
    all_rows = []

    for i in range(n_accounts):
        cust_id = f"CUST{100_000 + i}"
        subs    = random.sample(SUBSCRIPTIONS, k=random.randint(1, 4))
        all_rows.extend(_customer_transactions(cust_id, _random_name(), start, end, subs))

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["CustomerID", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Total rows      : {len(df):,}")
    print(f"  Subscriptions   : {df['SubscriptionFlag'].sum():,}  ({df['SubscriptionFlag'].mean()*100:.1f}%)")
    print(f"  Failed debits   : {(df['Status']=='FAILED').sum():,}")
    print(f"  Null - Amount   : {df['Amount'].isna().sum():,}  ({df['Amount'].isna().mean()*100:.2f}%)")
    print(f"  Null - Balance  : {df['Balance'].isna().sum():,}  ({df['Balance'].isna().mean()*100:.2f}%)")
    print(f"  Null - Desc     : {df['Description'].isna().sum():,}  ({df['Description'].isna().mean()*100:.2f}%)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")

    print(f"{'='*60}\n")
    return df


if __name__ == "__main__":
    df = generate_dataset(output_path="../data/transactions_raw.csv")
    print(df[["CustomerID","Date","Description","Amount","Balance","Merchant","SubscriptionFlag"]].head(10))
