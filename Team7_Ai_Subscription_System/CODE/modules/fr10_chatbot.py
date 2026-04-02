# """
# FR10 - GenAI RAG Chatbot
# AI Subscription & Auto-Debit Intelligence System
# Team 7 - Mansi & Samyak

# Architecture:
#   User Query
#       ↓
#   Intent Classifier  (keyword-based, instant)
#       ↓
#   Context Retriever  (pulls relevant rows from CSVs — the "RAG" step)
#       ↓
#   Phi-2 Local Model  (D:/phi2_local)  OR  fast rule-based fallback
#       ↓
#   Grounded Answer    (no hallucinations — every answer sourced from data)

# Phi-2 path: D:/phi2_local
# Fallback:   rule-based template answers (used if Phi-2 unavailable)
# Response target: <3 seconds
# """

# import os
# import re
# import time
# import pandas as pd

# # ── Phi-2 path (user's D: drive) ──────────────────────────────────────────────
# PHI2_PATH = r"D:/phi2_local"

# # ── Lazy-load Phi-2 once; fall back gracefully if not available ───────────────
# _phi2_model     = None
# _phi2_tokenizer = None
# _phi2_ok        = False


# def _load_phi2():
#     """Try to load Microsoft Phi-2 from D:/phi2_local. Silent fail → fallback."""
#     global _phi2_model, _phi2_tokenizer, _phi2_ok
#     if _phi2_ok:
#         return True
#     try:
#         import torch
#         from transformers import AutoTokenizer, AutoModelForCausalLM
#         print("[Chatbot] Loading Phi-2 from", PHI2_PATH, "...")
#         _phi2_tokenizer = AutoTokenizer.from_pretrained(
#             PHI2_PATH, trust_remote_code=True, local_files_only=True
#         )
#         _phi2_model = AutoModelForCausalLM.from_pretrained(
#             PHI2_PATH,
#             trust_remote_code=True,
#             local_files_only=True,
#             torch_dtype=torch.float32,   # CPU-safe
#         )
#         _phi2_model.eval()
#         _phi2_ok = True
#         print("[Chatbot] ✅ Phi-2 loaded successfully.")
#         return True
#     except Exception as e:
#         print(f"[Chatbot] ⚠️  Phi-2 not available ({e}). Using rule-based fallback.")
#         _phi2_ok = False
#         return False


# def _phi2_generate(prompt: str, max_new_tokens: int = 120) -> str:
#     """Run one forward pass through Phi-2 and return generated text."""
#     import torch
#     inputs = _phi2_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         output = _phi2_model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,          # greedy — deterministic & fast
#             temperature=1.0,
#             pad_token_id=_phi2_tokenizer.eos_token_id,
#         )
#     # Strip the prompt echo and return only the new tokens
#     generated = _phi2_tokenizer.decode(output[0], skip_special_tokens=True)
#     # Phi-2 echoes the prompt — remove it
#     if prompt.strip() in generated:
#         generated = generated[len(prompt.strip()):].strip()
#     return generated.strip()


# # ══════════════════════════════════════════════════════════════════════════════
# # INTENT DETECTION  (fast keyword match — runs before any LLM call)
# # ══════════════════════════════════════════════════════════════════════════════
# INTENTS = {
#     "count_subs":      ["how many", "count", "number of", "total subscriptions", "active sub"],
#     "total_spend":     ["spend", "spending", "cost", "monthly spend", "total spend", "how much"],
#     "risk_subs":       ["risk", "at risk", "fail", "failure", "which.*risk", "danger"],
#     "unused_subs":     ["not using", "unused", "inactive", "never use", "cancel"],
#     "next_debit":      ["next debit", "upcoming", "when.*debit", "next payment", "due date", "when.*charge"],
#     "top_merchants":   ["merchant", "vendor", "service", "which service", "top merchant"],
#     "balance":         ["balance", "account balance", "how much.*account", "funds"],
#     "high_risk_list":  ["high risk", "high-risk", "list.*high", "who.*high"],
#     "summary":         ["summary", "overview", "give me", "tell me about", "what.*subscription"],
#     "help":            ["help", "what can you", "how to use", "capabilities", "example"],
# }


# def detect_intent(query: str) -> str:
#     q = query.lower().strip()
#     for intent, patterns in INTENTS.items():
#         for pat in patterns:
#             if re.search(pat, q):
#                 return intent
#     return "general"


# # ══════════════════════════════════════════════════════════════════════════════
# # CONTEXT RETRIEVER  (the RAG step — pull relevant data rows)
# # ══════════════════════════════════════════════════════════════════════════════
# def _extract_customer(query: str) -> str | None:
#     """Pull CUSTXXXXXX from query if present."""
#     m = re.search(r"(CUST\d+)", query.upper())
#     return m.group(1) if m else None


# def retrieve_context(query: str, data: dict, customer_id: str | None = None) -> dict:
#     """
#     Pull the minimal relevant rows from data CSVs.
#     Returns a flat dict of facts used to ground the answer.
#     """
#     ctx = {}
#     cid = customer_id or _extract_customer(query)

#     insights = data.get("insights", pd.DataFrame())
#     risk     = data.get("risk",     pd.DataFrame())
#     pred     = data.get("pred",     pd.DataFrame())
#     summary  = data.get("summary",  pd.DataFrame())

#     # ── Filter to customer if specified ──
#     def filt(df):
#         if df.empty or cid is None or "CustomerID" not in df.columns:
#             return df
#         r = df[df["CustomerID"] == cid]
#         return r if not r.empty else df      # fall back to all if no match

#     ins_f  = filt(insights)
#     risk_f = filt(risk)
#     pred_f = filt(pred)
#     sum_f  = filt(summary)

#     # ── Core aggregates ──
#     if not ins_f.empty:
#         ctx["total_customers"]  = len(insights)
#         ctx["avg_active_subs"]  = round(float(ins_f["Active_Subscriptions"].mean()), 1)
#         ctx["avg_monthly_spend"]= round(float(ins_f["Total_Monthly_Spend"].mean()), 2)
#         ctx["max_monthly_spend"]= round(float(ins_f["Total_Monthly_Spend"].max()), 2)
#         ctx["customer_id"]      = cid or "all"

#         if cid:
#             row = ins_f.iloc[0]
#             ctx["active_subs"]      = int(row.get("Active_Subscriptions", 0))
#             ctx["monthly_spend"]    = round(float(row.get("Total_Monthly_Spend", 0)), 2)
#             ctx["upcoming_total"]   = round(float(row.get("Upcoming_Total", 0)), 2)
#             ctx["next_debit_date"]  = str(row.get("Next_Debit_Date", "N/A"))
#             ctx["balance"]          = round(float(row.get("Current_Balance", 0)), 2)
#             ctx["risk_level"]       = str(row.get("Risk_Level", "N/A"))
#             ctx["top_merchants"]    = str(row.get("Top_Merchants", "N/A"))
#             ctx["insight_msg"]      = str(row.get("FR8_Message", ""))

#     # ── Risk aggregates ──
#     if not risk_f.empty:
#         ctx["high_risk_count"]  = int((risk_f["Risk_Level"] == "High").sum())
#         ctx["medium_risk_count"]= int((risk_f["Risk_Level"] == "Medium").sum())
#         ctx["low_risk_count"]   = int((risk_f["Risk_Level"] == "Low").sum())

#         if cid and not risk_f.empty:
#             rrow = risk_f.iloc[0]
#             ctx["risk_score"]   = float(rrow.get("Risk_Score", 0))
#             ctx["risk_reason"]  = str(rrow.get("Risk_Reason", ""))
#             ctx["failed_debits"]= int(rrow.get("Failed_Debits", 0))
#             ctx["fail_rate"]    = round(float(rrow.get("Failed_Debit_Rate", 0)) * 100, 1)

#         # Top 5 high-risk customers
#         top_hr = risk[risk["Risk_Level"] == "High"].sort_values(
#             "Risk_Score", ascending=False
#         ).head(5)
#         ctx["top_high_risk"] = top_hr[["CustomerID", "Risk_Score", "Risk_Reason"]].to_dict("records") \
#                                if not top_hr.empty else []

#     # ── Prediction aggregates ──
#     if not pred_f.empty:
#         ctx["total_predictions"] = len(pred)
#         upcoming = pred_f.sort_values("Next_Debit_Date").head(5)
#         ctx["upcoming_debits"] = upcoming[
#             ["CustomerID", "Merchant", "Next_Debit_Date", "Predicted_Amount"]
#         ].to_dict("records") if not upcoming.empty else []

#     # ── Merchant aggregates ──
#     if not sum_f.empty:
#         top_m = sum_f.groupby("Merchant")["Occurrences"].sum() \
#                      .sort_values(ascending=False).head(5)
#         ctx["top_merchants_overall"] = top_m.to_dict()

#         # Unused: subscriptions seen <3 times (might be cancelled)
#         unused = sum_f[sum_f["Occurrences"] <= 2][["CustomerID", "Merchant", "Occurrences"]] \
#                  .head(5).to_dict("records")
#         ctx["unused_subs"] = unused

#     return ctx


# # ══════════════════════════════════════════════════════════════════════════════
# # RULE-BASED ANSWERER  (instant, no LLM, grounded in ctx)
# # ══════════════════════════════════════════════════════════════════════════════
# def rule_answer(intent: str, ctx: dict, query: str) -> str:
#     cid = ctx.get("customer_id", "all")

#     if intent == "count_subs":
#         if cid != "all":
#             n = ctx.get("active_subs", "N/A")
#             return (f"**{cid}** has **{n} active subscription(s)**.\n"
#                     f"Monthly spend: ₹{ctx.get('monthly_spend', 0):,.2f}\n"
#                     f"Top services: {ctx.get('top_merchants', 'N/A')}")
#         avg = ctx.get("avg_active_subs", "N/A")
#         return (f"Across all **{ctx.get('total_customers', '?')} customers**, "
#                 f"the average number of active subscriptions is **{avg}**.")

#     if intent == "total_spend":
#         if cid != "all":
#             spend = ctx.get("monthly_spend", 0)
#             return (f"**{cid}** has a total monthly subscription spend of "
#                     f"**₹{spend:,.2f}**.\n"
#                     f"Upcoming debit total: ₹{ctx.get('upcoming_total', 0):,.2f}\n"
#                     f"Next debit date: {ctx.get('next_debit_date', 'N/A')}")
#         avg  = ctx.get("avg_monthly_spend", 0)
#         mx   = ctx.get("max_monthly_spend", 0)
#         return (f"Average monthly subscription spend across all customers: **₹{avg:,.2f}**\n"
#                 f"Highest spender: **₹{mx:,.2f}** per month.")

#     if intent == "risk_subs":
#         high = ctx.get("high_risk_count", 0)
#         med  = ctx.get("medium_risk_count", 0)
#         low  = ctx.get("low_risk_count", 0)
#         if cid != "all":
#             lvl    = ctx.get("risk_level", "N/A")
#             score  = ctx.get("risk_score", 0)
#             reason = ctx.get("risk_reason", "")
#             icon   = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(lvl, "⚪")
#             return (f"{icon} **{cid}** is at **{lvl} Risk** (score: {score:.4f})\n"
#                     f"Reason: {reason}\n"
#                     f"Failed debits: {ctx.get('failed_debits', 0)} | "
#                     f"Failure rate: {ctx.get('fail_rate', 0):.1f}%")
#         lines = [f"🔴 **High Risk:** {high} customers",
#                  f"🟡 **Medium Risk:** {med} customers",
#                  f"🟢 **Low Risk:** {low} customers"]
#         top = ctx.get("top_high_risk", [])
#         if top:
#             lines.append("\n**Top High-Risk Accounts:**")
#             for r in top:
#                 lines.append(f"• {r['CustomerID']} — score {r['Risk_Score']:.4f} | {r['Risk_Reason'][:60]}")
#         return "\n".join(lines)

#     if intent == "unused_subs":
#         unused = ctx.get("unused_subs", [])
#         if not unused:
#             return "No potentially unused subscriptions found in the dataset."
#         lines = ["Subscriptions with very few occurrences (≤2) — possibly unused or cancelled:"]
#         for u in unused:
#             lines.append(f"• {u.get('CustomerID')} — **{u.get('Merchant')}** "
#                          f"({u.get('Occurrences')} occurrence(s))")
#         lines.append("\nℹ️ A subscription appearing <3 times may have been cancelled or is very new.")
#         return "\n".join(lines)

#     if intent == "next_debit":
#         if cid != "all":
#             return (f"**{cid}'s** next predicted debit:\n"
#                     f"📅 Date: **{ctx.get('next_debit_date', 'N/A')}**\n"
#                     f"💰 Amount: ₹{ctx.get('upcoming_total', 0):,.2f}\n"
#                     f"⚠️ Risk level: {ctx.get('risk_level', 'N/A')}")
#         debits = ctx.get("upcoming_debits", [])
#         if not debits:
#             return "No upcoming debit predictions available."
#         lines = ["**Upcoming Debits (next 5):**"]
#         for d in debits:
#             lines.append(f"• {d.get('CustomerID')} — {d.get('Merchant')} "
#                          f"₹{d.get('Predicted_Amount', 0):,.2f} "
#                          f"on {str(d.get('Next_Debit_Date', ''))[:10]}")
#         return "\n".join(lines)

#     if intent == "top_merchants":
#         top = ctx.get("top_merchants_overall", {})
#         if not top:
#             return "Merchant data not available."
#         lines = ["**Top Merchants by Subscription Volume:**"]
#         for i, (m, cnt) in enumerate(top.items(), 1):
#             lines.append(f"{i}. **{m}** — {int(cnt):,} occurrences")
#         return "\n".join(lines)

#     if intent == "balance":
#         if cid != "all":
#             bal  = ctx.get("balance", 0)
#             upco = ctx.get("upcoming_total", 0)
#             gap  = bal - upco
#             status = "✅ Sufficient" if gap > 0 else "⚠️ Insufficient"
#             return (f"**{cid}** — Account Balance: **₹{bal:,.2f}**\n"
#                     f"Upcoming Debit Total: ₹{upco:,.2f}\n"
#                     f"After debit: ₹{gap:,.2f} — {status}")
#         return "Please specify a CustomerID (e.g. CUST100001) to check balance."

#     if intent == "high_risk_list":
#         top = ctx.get("top_high_risk", [])
#         high = ctx.get("high_risk_count", 0)
#         lines = [f"**{high} customers are at High Risk.** Top accounts:"]
#         for r in top:
#             lines.append(f"• **{r['CustomerID']}** — Risk Score: {r['Risk_Score']:.4f}\n"
#                          f"  ↳ {r['Risk_Reason'][:80]}")
#         return "\n".join(lines) if top else "No high-risk customers found."

#     if intent == "summary":
#         if cid != "all":
#             msg = ctx.get("insight_msg", "")
#             if msg:
#                 return f"**Summary for {cid}:**\n{msg}"
#         total = ctx.get("total_customers", "?")
#         avg_s = ctx.get("avg_active_subs", "?")
#         avg_m = ctx.get("avg_monthly_spend", 0)
#         high  = ctx.get("high_risk_count", 0)
#         return (f"**SubIntel Dashboard Summary**\n"
#                 f"• Total customers: **{total}**\n"
#                 f"• Avg active subscriptions: **{avg_s}**\n"
#                 f"• Avg monthly spend: **₹{avg_m:,.2f}**\n"
#                 f"• High-risk customers: **{high}**\n"
#                 f"• Total predictions made: **{ctx.get('total_predictions', '?')}**")

#     if intent == "help":
#         return (
#             "**SubIntel Chatbot — What I can answer:**\n"
#             "• *How many active subscriptions do I have?* → Count with details\n"
#             "• *What is my total monthly spend?* → Amount shown\n"
#             "• *Which subscriptions are at risk?* → Risk levels listed\n"
#             "• *Which subscriptions am I not using?* → Unused flagged\n"
#             "• *When is my next debit?* → Date + amount predicted\n"
#             "• *Who are the top merchants?* → Ranked by volume\n"
#             "• *Summary for CUST100001* → Full customer overview\n\n"
#             "💡 **Tip:** Include a CustomerID (e.g. CUST100001) for customer-specific answers."
#         )

#     # Fallback general
#     total = ctx.get("total_customers", "?")
#     high  = ctx.get("high_risk_count", 0)
#     avg_m = ctx.get("avg_monthly_spend", 0)
#     return (f"I can answer questions about your subscriptions, risk levels, upcoming debits, "
#             f"and monthly spend.\n\nQuick stats: **{total}** customers tracked | "
#             f"**{high}** at high risk | Avg spend **₹{avg_m:,.2f}/mo**\n\n"
#             f"Try: *'How many subscriptions does CUST100001 have?'*")


# # ══════════════════════════════════════════════════════════════════════════════
# # PHI-2 ANSWERER  (used when model loads successfully)
# # ══════════════════════════════════════════════════════════════════════════════
# def phi2_answer(intent: str, ctx: dict, query: str) -> str:
#     """Build a grounded prompt and run Phi-2. Falls back to rule if it times out."""
#     # Build a compact context string from retrieved facts
#     ctx_lines = []
#     for k, v in ctx.items():
#         if k in ("top_high_risk", "upcoming_debits", "unused_subs", "top_merchants_overall"):
#             continue  # these are lists — skip in prompt, handled by rule layer
#         ctx_lines.append(f"{k}: {v}")
#     context_str = "\n".join(ctx_lines[:20])  # cap prompt size

#     prompt = (
#         "You are SubIntel, an AI assistant for a banking subscription intelligence system.\n"
#         "Answer the user's question ONLY using the data context below. "
#         "Be concise (2-4 lines). Do not invent numbers.\n\n"
#         f"DATA CONTEXT:\n{context_str}\n\n"
#         f"USER QUESTION: {query}\n\n"
#         "ANSWER:"
#     )

#     try:
#         t0  = time.time()
#         ans = _phi2_generate(prompt, max_new_tokens=100)
#         elapsed = time.time() - t0
#         # If Phi-2 took > 4 seconds or gave empty output, fall back
#         if elapsed > 4.0 or len(ans.strip()) < 5:
#             return rule_answer(intent, ctx, query)
#         return ans.strip()
#     except Exception:
#         return rule_answer(intent, ctx, query)


# # ══════════════════════════════════════════════════════════════════════════════
# # PUBLIC API  (called by dashboard)
# # ══════════════════════════════════════════════════════════════════════════════
# def init_chatbot():
#     """
#     Call once at startup. Tries to load Phi-2; sets global flag.
#     Returns True if Phi-2 loaded, False if using fallback.
#     """
#     return _load_phi2()


# def chat(query: str, data: dict, customer_id: str | None = None) -> dict:
#     """
#     Main entry point.
#     Returns:
#         {
#           "answer":   str,       # the grounded answer
#           "intent":   str,       # detected intent
#           "model":    str,       # "phi2" or "rule-based"
#           "latency_ms": int,
#         }
#     """
#     t0     = time.time()
#     intent = detect_intent(query)
#     ctx    = retrieve_context(query, data, customer_id)

#     if _phi2_ok:
#         answer = phi2_answer(intent, ctx, query)
#         model  = "phi2"
#     else:
#         answer = rule_answer(intent, ctx, query)
#         model  = "rule-based"

#     latency = int((time.time() - t0) * 1000)
#     return {
#         "answer":      answer,
#         "intent":      intent,
#         "model":       model,
#         "latency_ms":  latency,
#     }