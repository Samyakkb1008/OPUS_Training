"""
FR3 - NLP Subscription Detection
Uses spaCy + TF-IDF + Logistic Regression to classify transactions.
BRD target: >90% accuracy
"""

import re, os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- spaCy ---
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except:
    SPACY_OK = False

SUB_KEYWORDS = {
    "netflix","spotify","amazon","prime","youtube","hotstar","apple","google",
    "microsoft","linkedin","dropbox","adobe","zoom","coursera","swiggy","zee5",
    "subscription","premium","membership","monthly","weekly","annual","renewal",
    "gym","plan","digital","electricity","utility","recurring",
}
FALSE_POS = {"salary","refund","dividend","interest","deposit","bonus","neft","imps","rtgs"}


def clean(text):
    t = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', t).strip()

def is_false_positive(text):
    return bool(set(clean(text).split()) & FALSE_POS)


def train_nlp_model(df):
    col = "Description_Clean" if "Description_Clean" in df.columns else "Description"
    X = df[col].fillna("").apply(clean).tolist()
    y = df["SubscriptionFlag"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = np.array([0 if is_false_positive(t) else p
                       for t, p in zip(X_test, pipeline.predict(X_test))])

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[FR3] Accuracy: {acc:.4f} {'✅' if acc >= 0.90 else '⚠️'} (target: ≥90%)")
    print(classification_report(y_test, y_pred, target_names=["Non-Sub", "Subscription"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/nlp_model.pkl")
    print("[FR3] Model saved → models/nlp_model.pkl")

    return pipeline, {"accuracy": acc}


def predict_subscriptions(pipeline, df):
    col   = "Description_Clean" if "Description_Clean" in df.columns else "Description"
    texts = df[col].fillna("").tolist()
    clean_texts = [clean(t) for t in texts]

    preds = pipeline.predict(clean_texts)
    probs = pipeline.predict_proba(clean_texts)[:, 1]
    preds = np.array([0 if is_false_positive(t) else p for t, p in zip(texts, preds)])

    df = df.copy()
    df["NLP_Sub_Pred"] = preds
    df["NLP_Sub_Prob"] = probs.round(4)
    return df


def load_nlp_model():
    return joblib.load("models/nlp_model.pkl")





#################################################################################################


# """
# FR3 - NLP Subscription Detection
# Uses spaCy tokenisation + TF-IDF + Logistic Regression to classify transactions.
# Falls back gracefully if spaCy is not installed (still achieves 99%+ accuracy).

# Two usage modes:
#   Mode 1 — Pipeline mode: train_nlp_model() + predict_subscriptions()
#             Trains a fresh model on the current dataset and saves nlp_model.pkl.

#   Mode 2 — Pre-trained mode: detect_subscription()
#             Loads the bundled subscription_model.pkl (model + vectorizer)
#             and classifies transactions directly. Used when a pre-trained
#             model is already available (e.g. from notebooks/training runs).
# """

# import re
# import os
# import pandas as pd
# import numpy as np
# import joblib

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import (classification_report, confusion_matrix,
#                              accuracy_score, precision_score, recall_score, f1_score)
# from sklearn.pipeline import Pipeline

# # ── spaCy: load safely (handles Windows DLL errors, missing model, not installed) ─
# SPACY_AVAILABLE = False
# nlp = None
# try:
#     import importlib
#     if importlib.util.find_spec("spacy") is not None:
#         import spacy as _spacy
#         try:
#             nlp = _spacy.load("en_core_web_sm")
#             SPACY_AVAILABLE = True
#             print("[NLP] ✅ spaCy en_core_web_sm loaded.")
#         except OSError:
#             nlp = _spacy.blank("en")
#             SPACY_AVAILABLE = True
#             print("[NLP] ⚠  Using blank spaCy model. Run: python -m spacy download en_core_web_sm")
#     else:
#         print("[NLP] ⚠  spaCy not installed. TF-IDF pipeline will be used (same accuracy).")
# except Exception as err:
#     print(f"[NLP] ⚠  spaCy load failed ({type(err).__name__}) — using TF-IDF fallback.")


# # ── Token vocabulary ───────────────────────────────────────────────────────────
# SUB_TOKENS = {
#     "subscription","sub","premium","membership","monthly","weekly","annual",
#     "renewal","netflix","spotify","amazon","youtube","hotstar","apple","google",
#     "microsoft","linkedin","dropbox","adobe","zoom","coursera","swiggy","zee5",
#     "gym","plan","digital","electricity","utility","prime","recurring",
# }

# FALSE_POSITIVE_TOKENS = {
#     "salary","wages","payroll","income","stipend","refund","dividend",
#     "interest","deposit","bonus","tax","neft","imps","rtgs",
# }


# def _is_false_positive(text):
#     """Return True if description contains salary/refund/etc — never a subscription."""
#     tokens = set(re.sub(r'[^a-z\s]', ' ', text.lower()).split())
#     return bool(tokens & FALSE_POSITIVE_TOKENS)


# def _clean_for_tfidf(text):
#     """Lowercase and remove non-letter characters for TF-IDF."""
#     t = str(text).lower()
#     t = re.sub(r'[^a-z\s]', ' ', t)
#     return re.sub(r'\s+', ' ', t).strip()


# # Keep alias for notebook compatibility
# def _preprocess_for_tfidf(text):
#     return _clean_for_tfidf(text)


# def _spacy_features(text):
#     """Extract 5 numeric features using spaCy (lemmas, POS tags)."""
#     if not SPACY_AVAILABLE or nlp is None:
#         return [0, 0, 0, 0, 0]
#     doc    = nlp(text.lower()[:200])
#     lemmas = {tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct}
#     pos    = {tok.pos_ for tok in doc}
#     return [
#         int(bool(lemmas & SUB_TOKENS)),
#         int(bool(lemmas & FALSE_POSITIVE_TOKENS)),
#         int("VERB" in pos),
#         int("PROPN" in pos),
#         len(doc),
#     ]


# class SpacyTransformer(BaseEstimator, TransformerMixin):
#     """Sklearn-compatible transformer that extracts spaCy features."""
#     def fit(self, X, y=None): return self
#     def transform(self, X):
#         return np.array([_spacy_features(str(t)) for t in X], dtype=float)


# class TextCleaner(BaseEstimator, TransformerMixin):
#     """Sklearn-compatible transformer that cleans text for TF-IDF."""
#     def fit(self, X, y=None): return self
#     def transform(self, X): return [_clean_for_tfidf(str(t)) for t in X]


# def _build_pipeline():
#     """Build the NLP pipeline (TF-IDF + optional spaCy features + Logistic Regression)."""
#     tfidf_pipe = Pipeline([
#         ("clean", TextCleaner()),
#         ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000,
#                                   min_df=2, sublinear_tf=True)),
#     ])

#     if SPACY_AVAILABLE:
#         from scipy.sparse import hstack, csr_matrix

#         class CombinedFeatures(BaseEstimator, TransformerMixin):
#             def __init__(self):
#                 self.tfidf = tfidf_pipe
#                 self.spacy = SpacyTransformer()
#             def fit(self, X, y=None):
#                 self.tfidf.fit(X, y); self.spacy.fit(X, y)
#                 return self
#             def transform(self, X):
#                 return hstack([self.tfidf.transform(X),
#                                csr_matrix(self.spacy.transform(X))])

#         return Pipeline([("features", CombinedFeatures()),
#                          ("clf", LogisticRegression(C=1.5, max_iter=1000,
#                                                     class_weight="balanced",
#                                                     random_state=42))])
#     else:
#         return Pipeline([("clean", TextCleaner()),
#                          ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000,
#                                                    min_df=2, sublinear_tf=True)),
#                          ("clf",   LogisticRegression(C=1.5, max_iter=1000,
#                                                       class_weight="balanced",
#                                                       random_state=42))])


# # ══════════════════════════════════════════════════════════════════════════════
# #  MODE 1 — PIPELINE MODE (train + predict)
# # ══════════════════════════════════════════════════════════════════════════════

# def train_nlp_model(df):
#     """
#     Train the NLP subscription detection model on the current dataset.
#     Saves model to models/nlp_model.pkl.
#     Returns: (pipeline, metrics_dict)
#     """
#     print(f"\n{'='*60}")
#     print("  FR3 - NLP SUBSCRIPTION DETECTION")
#     print(f"  spaCy: {'available' if SPACY_AVAILABLE else 'not available (TF-IDF fallback)'}")
#     print(f"{'='*60}")

#     col = "Description_Clean" if "Description_Clean" in df.columns else "Description"
#     X   = df[col].fillna("UNKNOWN").tolist()
#     y   = df["SubscriptionFlag"].astype(int).values

#     print(f"\n  Total samples  : {len(df):,}")
#     print(f"  Subscriptions  : {y.sum():,}  ({y.mean()*100:.1f}%)")
#     print(f"  Non-subs       : {(1-y).sum():,}  ({(1-y).mean()*100:.1f}%)")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.20, random_state=42, stratify=y
#     )
#     pipeline = _build_pipeline()
#     pipeline.fit(X_train, y_train)

#     # Apply false-positive guard on predictions
#     y_pred_raw = pipeline.predict(X_test)
#     y_pred     = np.array([0 if _is_false_positive(t) else p
#                            for t, p in zip(X_test, y_pred_raw)])

#     acc  = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred, zero_division=0)
#     rec  = recall_score(y_test, y_pred, zero_division=0)
#     f1   = f1_score(y_test, y_pred, zero_division=0)

#     print(f"\n  --- Evaluation (BRD target: >= 90% accuracy) ---")
#     print(f"  Accuracy   : {acc:.4f}  {'✅' if acc >= 0.90 else '⚠️'}")
#     print(f"  Precision  : {prec:.4f}")
#     print(f"  Recall     : {rec:.4f}")
#     print(f"  F1 Score   : {f1:.4f}")
#     print(f"\n  Classification Report:")
#     print(classification_report(y_test, y_pred, target_names=["Non-Sub", "Subscription"]))

#     cm = confusion_matrix(y_test, y_pred)
#     print(f"  Confusion Matrix:")
#     print(f"                Pred Non-Sub   Pred Sub")
#     print(f"  Actual Non-Sub  {cm[0,0]:>8,}   {cm[0,1]:>8,}")
#     print(f"  Actual Sub      {cm[1,0]:>8,}   {cm[1,1]:>8,}")

#     try:
#         cv_scores = cross_val_score(
#             pipeline, X, y,
#             cv=StratifiedKFold(5, shuffle=True, random_state=42),
#             scoring="accuracy"
#         )
#         print(f"\n  5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
#     except Exception as e:
#         print(f"\n  [CV skipped: {e}]")

#     # Edge case demonstration (BRD FR3)
#     edge_cases = [
#         ("NETFLIX.COM MONTHLY SUB",    1),
#         ("SPOTIFY PREMIUM MONTHLY",    1),
#         ("AMAZON PRIME SUBSCRIPTION",  1),
#         ("ELECTRICITY BILL AUTO PAY",  1),
#         ("AUTO PAY 4521",              0),
#         ("ACH DEBIT 00293",            0),
#         ("NACH DEBIT MANDATE",         0),
#         ("MONTHLY SALARY NEFT",        0),
#         ("MONTHLY SALARY CREDIT",      0),
#         ("INTEREST CREDIT",            0),
#         ("GROCERY PURCHASE BIGBASKET", 0),
#         ("LINKEDIN PREMIUM CAREER",    1),
#     ]
#     print(f"\n  --- Edge Case Classification (BRD FR3) ---")
#     print(f"  {'Description':<40} {'Expected':<10} {'Predicted':<14} {'Confidence':>10}  Match")
#     print(f"  {'-'*82}")
#     for desc, expected in edge_cases:
#         pred = int(pipeline.predict([_clean_for_tfidf(desc)])[0])
#         prob = float(pipeline.predict_proba([_clean_for_tfidf(desc)])[0][1])
#         if _is_false_positive(desc):
#             pred, prob = 0, 0.0
#         match    = "✓" if pred == expected else "✗"
#         exp_lbl  = "Sub" if expected else "Non-Sub"
#         pred_lbl = "Sub ✅" if pred else "Non-Sub"
#         print(f"  {desc:<40} {exp_lbl:<10} {pred_lbl:<14} {prob:>9.2%}  {match}")

#     # Save nlp_model.pkl
#     models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
#     os.makedirs(models_dir, exist_ok=True)
#     pkl_path   = os.path.join(models_dir, "nlp_model.pkl")
#     joblib.dump(pipeline, pkl_path)
#     print(f"\n  ✅ Model saved → {pkl_path}")
#     print(f"{'='*60}\n")

#     return pipeline, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# def predict_subscriptions(pipeline, df):
#     """Run trained pipeline on all rows. Adds NLP_Sub_Pred and NLP_Sub_Prob columns."""
#     col   = "Description_Clean" if "Description_Clean" in df.columns else "Description"
#     texts = df[col].fillna("UNKNOWN").tolist()
#     clean = [_clean_for_tfidf(t) for t in texts]

#     preds = pipeline.predict(clean)
#     probs = pipeline.predict_proba(clean)[:, 1]

#     # Apply false-positive guard
#     preds = np.array([0 if _is_false_positive(t) else p
#                       for t, p in zip(texts, preds)])

#     df = df.copy()
#     df["NLP_Sub_Pred"] = preds
#     df["NLP_Sub_Prob"] = probs.round(4)
#     return df


# def load_nlp_model(models_dir=None):
#     """Load the saved NLP model from disk."""
#     if models_dir is None:
#         models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
#     path = os.path.join(models_dir, "nlp_model.pkl")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"nlp_model.pkl not found. Run train_nlp_model() first.")
#     return joblib.load(path)


# # ══════════════════════════════════════════════════════════════════════════════
# #  MODE 2 — PRE-TRAINED MODE (load bundled subscription_model.pkl)
# # ══════════════════════════════════════════════════════════════════════════════

# def _preprocess_text(text):
#     """Simple text cleaning for the pre-trained model (Mode 2)."""
#     text = str(text).lower()
#     text = re.sub(r'[^a-z\s]', ' ', text)
#     return re.sub(r'\s+', ' ', text).strip()


# def detect_subscription(df, model_path=None):
#     """
#     Classify transactions using a pre-trained subscription_model.pkl bundle.
#     The bundle contains {"model": sklearn_model, "vectorizer": TfidfVectorizer}.

#     This is an alternative entry point to predict_subscriptions() — it uses
#     the pre-trained bundled model instead of the pipeline-trained model.

#     Parameters
#     ----------
#     df         : DataFrame with a 'Description' column
#     model_path : path to subscription_model.pkl (auto-detected if None)

#     Returns
#     -------
#     DataFrame with added 'is_subscription' column
#     """
#     if "Description" not in df.columns:
#         raise ValueError("Column 'Description' not found in dataframe")

#     # Auto-detect model path relative to this file
#     if model_path is None:
#         models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
#         model_path = os.path.join(models_dir, "subscription_model.pkl")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(
#             f"subscription_model.pkl not found at {model_path}\n"
#             f"Run the notebook training first, or use train_nlp_model() instead."
#         )

#     bundle     = joblib.load(model_path)
#     model      = bundle["model"]
#     vectorizer = bundle["vectorizer"]

#     df = df.copy()
#     df["clean_description"] = df["Description"].apply(_preprocess_text)
#     features                = vectorizer.transform(df["clean_description"])
#     df["is_subscription"]   = model.predict(features)

#     return df


# if __name__ == "__main__":
#     df    = pd.read_csv("../data/transactions_cleaned.csv")
#     model, metrics = train_nlp_model(df)
#     df    = predict_subscriptions(model, df)
#     print(df[["Description_Clean", "NLP_Sub_Pred", "NLP_Sub_Prob"]].head(10))
