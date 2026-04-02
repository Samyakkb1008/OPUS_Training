"""
FR7 - GenAI Alert Generator
Tries Microsoft Phi-2 first. Falls back to structured rule-based alerts.
BRD FR7 format: Upcoming Debit: Netflix Rs499, Risk: Medium (0.53), Suggestion: Add funds before 2 days
"""

import os
import pandas as pd
from datetime import date
from collections import Counter

PHI2_PATH  = os.environ.get("PHI2_MODEL_PATH", "D:\\phi2_local")
_phi2_pipe = None


def _load_phi2():
    global _phi2_pipe
    if _phi2_pipe is not None:
        return _phi2_pipe
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[FR7] Loading Phi-2 from '{PHI2_PATH}' ...")
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(PHI2_PATH, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
            PHI2_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        _phi2_pipe = (tokenizer, model, device)
        print(f"[FR7] Phi-2 ready on {device.upper()}")
    except Exception as e:
        print(f"[FR7] Phi-2 unavailable ({e}) -- using structured rule-based fallback.")
        _phi2_pipe = "FALLBACK"
    return _phi2_pipe


def _generate_phi2(prompt):
    pipe = _load_phi2()
    if pipe == "FALLBACK":
        return None
    try:
        import torch
        tokenizer, model, device = pipe
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=120, temperature=0.3,
                do_sample=True, top_p=0.9, repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[FR7] Generation error: {e}")
        return None


def _build_phi2_prompt(merchant, amount, due_date, risk_level, balance, upcoming_total, days_left):
    shortfall = max(0, upcoming_total - balance)
    if shortfall > 0:
        suggestion = f"Add funds Rs{shortfall:,.0f} before {days_left} days"
    else:
        suggestion = f"Maintain balance above Rs{upcoming_total*1.1:,.0f}"

    return (
        f"Instruct: Generate a one-line bank alert in EXACTLY this format:\n"
        f"Upcoming Debit: [Merchant] Rs[Amount], Risk: [Level] ([reason]), Suggestion: [Action]\n\n"
        f"Data: Merchant={merchant}, Amount=Rs{amount:,.2f}, Due={due_date} (in {days_left} days), "
        f"Risk={risk_level}, Balance=Rs{balance:,.0f}, Upcoming=Rs{upcoming_total:,.0f}\n\n"
        f"Output ONLY the one alert line:\nUpcoming Debit:"
    )


def _rule_based_alert_line(merchant, amount, due_date, risk_level, risk_score, balance,
                            upcoming_total, reason, days_left):
    """Generate BRD FR7 format alert line using rule-based structured generation."""
    shortfall = max(0, upcoming_total - balance)

    # Suggestion based on risk level
    if risk_level == "High":
        if shortfall > 0:
            suggestion = f"Top up Rs{shortfall:,.0f} immediately. Debit due in {days_left} day(s)"
        else:
            suggestion = f"Monitor balance closely. Multiple debits due"
    elif risk_level == "Medium":
        suggestion = f"Maintain Rs{upcoming_total*1.1:,.0f} before {due_date}"
    else:
        suggestion = "Account balance sufficient. No action needed"

    # BRD FR7 exact format
    return (
        f"Upcoming Debit: {merchant} Rs{amount:,.2f} | "
        f"Risk: {risk_level} ({risk_score:.2f}) | "
        f"Suggestion: {suggestion}"
    )


def _format_alert_block(cust_id, alert_lines, risk_info, subs_info, insights_info, same_date_warn):
    """Format full structured alert block for alerts.txt."""
    level  = risk_info["level"]
    score  = risk_info["score"]
    reason = risk_info["reason"]
    bal    = risk_info["balance"]
    total  = risk_info["upcoming_total"]
    emoji  = {"High": "RED", "Medium": "YELLOW", "Low": "GREEN"}.get(level, "")

    lines = [
        "=" * 60,
        f"  CUSTOMER : {cust_id}",
        "=" * 60,
        "",
        f"[FR7] GENAI ALERTS (Microsoft Phi-2 / Structured)",
        "-" * 60,
    ]

    for line in alert_lines:
        lines.append(f"  * {line}")

    if same_date_warn:
        lines += ["", f"  WARNING: {same_date_warn}"]

    lines += [
        "",
        f"[{emoji}] RISK ASSESSMENT (FR6)",
        "-" * 60,
        f"  Risk Level  : {level}",
        f"  Risk Score  : {score:.4f}",
        f"  Reason      : {reason}",
        f"  Balance     : Rs{bal:,.2f}",
        f"  Upcoming    : Rs{total:,.2f}",
        "",
        "SUBSCRIPTION SUMMARY (FR8)",
        "-" * 60,
        f"  Active Subscriptions : {insights_info.get('count', len(subs_info))}",
        f"  Total Monthly Spend  : Rs{insights_info.get('total_spend', total):,.2f}",
        "",
        "VALIDATION",
        "  Alert types: Upcoming debit | Risk assessment | Actionable suggestion",
        "  Content: Merchant, Amount, Date, Risk Level, Score, Reason",
        "  Format: Consistent BRD FR7 structured format",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


def generate_alerts(risk_df, pred_df, insights_df=None, customer_df=None, top_n=None, use_phi2=True):
    """
    Generate FR7 structured alerts for ALL customers with upcoming debits.
    Tries Phi-2 first; falls back to structured rule-based generation.
    """
    print("\n[FR7] Generating alerts...")

    phi2_ready = use_phi2 and (_load_phi2() != "FALLBACK")
    print(f"  Mode: {'Phi-2 LLM' if phi2_ready else 'Structured rule-based (Phi-2 unavailable)'}")

    # Target: ALL customers with upcoming debits (any risk level)
    target = (
        risk_df[risk_df["Upcoming_Total_Debit"] > 0]
        .sort_values("Risk_Score", ascending=False)
    )

    insights_map = {}
    if insights_df is not None and not insights_df.empty:
        for _, r in insights_df.iterrows():
            insights_map[r["CustomerID"]] = {
                "count":       r.get("Active_Subscriptions", 0),
                "total_spend": r.get("Total_Monthly_Spend", 0),
            }

    alerts  = []
    printed = 0

    for _, rrow in target.iterrows():
        cust_id    = rrow["CustomerID"]
        cust_preds = pred_df[pred_df["CustomerID"] == cust_id]
        if cust_preds.empty:
            continue

        risk_info = {
            "level":          rrow["Risk_Level"],
            "score":          float(rrow["Risk_Score"]),
            "reason":         rrow["Risk_Reason"],
            "balance":        float(rrow["Current_Balance"]),
            "upcoming_total": float(rrow["Upcoming_Total_Debit"]),
        }
        insights_info = insights_map.get(cust_id, {
            "count":       int(rrow.get("Subscription_Count", 0)),
            "total_spend": float(rrow["Upcoming_Total_Debit"]),
        })

        subs_info = []
        for _, r in cust_preds.iterrows():
            due  = pd.Timestamp(str(r["Next_Debit_Date"])).date()
            days = (due - date.today()).days
            subs_info.append({
                "sub":      r["Subscription"],
                "merchant": r.get("Merchant", r["Subscription"].split()[0].title()),
                "amount":   float(r["Predicted_Amount"]),
                "due_date": due,
                "days":     days,
            })

        # Detect same-date subscriptions (BRD scenario)
        date_counts    = Counter(str(s["due_date"]) for s in subs_info)
        same_date_warn = ""
        same_days      = [d for d, cnt in date_counts.items() if cnt > 1]
        if same_days:
            same_date_warn = f"Multiple subscriptions due on {', '.join(same_days)} -- ensure sufficient balance."

        # Generate alert lines
        alert_lines = []
        for s in subs_info:
            if phi2_ready:
                prompt = _build_phi2_prompt(
                    s["merchant"], s["amount"], s["due_date"],
                    risk_info["level"], risk_info["balance"],
                    risk_info["upcoming_total"], s["days"]
                )
                raw = _generate_phi2(prompt)
                if raw:
                    first = raw.split("\n")[0].strip()
                    line  = first if first.startswith("Upcoming Debit:") else f"Upcoming Debit: {first}"
                    alert_lines.append(line)
                    continue

            # Rule-based fallback
            line = _rule_based_alert_line(
                s["merchant"], s["amount"], s["due_date"],
                risk_info["level"], risk_info["score"],
                risk_info["balance"], risk_info["upcoming_total"],
                risk_info["reason"], s["days"]
            )
            alert_lines.append(line)

        if not alert_lines:
            continue

        alert_text = _format_alert_block(
            cust_id, alert_lines, risk_info, subs_info, insights_info, same_date_warn
        )

        alerts.append({
            "customer_id":   cust_id,
            "risk_level":    risk_info["level"],
            "risk_score":    risk_info["score"],
            "alert_text":    alert_text,
        })

        if printed < 3:
            print(alert_text)
            printed += 1

    print(f"  Total alerts generated : {len(alerts)}")
    return alerts
