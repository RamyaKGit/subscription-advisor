"""
retrain.py — Weekly retraining script

Run this once a week manually or via a scheduler:
    python retrain.py

What it does:
1. Reads pending_feedback.csv — predictions the user agreed/disagreed with
2. Only uses feedback older than 7 days (gives user time to act on advice)
3. AGREE  → keeps model's prediction as the label
   DISAGREE → flips to the opposite label
4. Appends new labeled rows to features.csv
5. Retrains the model on the combined dataset
6. Saves updated model.pkl
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR     = os.path.dirname(__file__)
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODEL_DIR    = os.path.join(BASE_DIR, "model")

FEATURES_FILE        = os.path.join(DATA_DIR, "features.csv")
PENDING_FEEDBACK_FILE = os.path.join(DATA_DIR, "pending_feedback.csv")
FEEDBACK_LOG_FILE    = os.path.join(DATA_DIR, "feedback_log.csv")

FEATURE_COLS = [
    "monthly_cost",
    "avg_hrs_per_month",
    "recent_hours",
    "recent_sessions",
    "recent_active_months",
    "days_since_last_use",
    "cost_per_hour",
    "usage_trend",
    "days_to_renewal",
    "has_free_tier",
    "has_cheaper_tier",
    "cheaper_saving",
    "category_encoded",
    "billing_encoded",
]

LABEL_ORDER = ["CLOSE", "REVIEW", "OPEN"]

def flip_label(label):
    """Flip DISAGREE labels to opposite class."""
    flips = {
        "CLOSE":  "REVIEW",   # CLOSE → REVIEW (not OPEN, too far)
        "REVIEW": "OPEN",     # REVIEW → OPEN
        "OPEN":   "REVIEW",   # OPEN → REVIEW
    }
    return flips.get(label, label)

def load_pending_feedback():
    """Load feedback that is at least 7 days old and has been filled in."""
    if not os.path.exists(PENDING_FEEDBACK_FILE):
        print("No pending feedback file found.")
        return pd.DataFrame()

    df = pd.read_csv(PENDING_FEEDBACK_FILE)

    # Only rows with actual feedback
    df = df[df["user_feedback"].isin(["AGREE", "DISAGREE"])]
    if df.empty:
        print("No feedback given yet.")
        return pd.DataFrame()

    # Only feedback older than 7 days
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    cutoff = datetime.today() - timedelta(days=7)
    df = df[df["prediction_date"] <= cutoff]

    if df.empty:
        print("No feedback older than 7 days yet. Run again next week.")
        return pd.DataFrame()

    print(f"Found {len(df)} feedback rows older than 7 days.")
    return df

def convert_feedback_to_labels(feedback_df):
    """
    Convert user feedback into labeled training rows.
    AGREE    → keep model's prediction as target
    DISAGREE → flip to adjacent label
    """
    rows = []
    for _, row in feedback_df.iterrows():
        target = row["prediction"] if row["user_feedback"] == "AGREE" else flip_label(row["prediction"])
        new_row = {
            "user_name":            row["user_name"],
            "subscription_name":    row["subscription_name"],
            "category":             row["category"],
            "billing_cycle":        row["billing_cycle"],
            "monthly_cost":         row["monthly_cost"],
            "total_hours":          row.get("total_hours", 0),
            "total_sessions":       row.get("total_sessions", 0),
            "avg_hrs_per_month":    row["avg_hrs_per_month"],
            "recent_hours":         row.get("recent_hours", 0),
            "recent_sessions":      row.get("recent_sessions", 0),
            "recent_active_months": row.get("recent_active_months", 0),
            "days_since_last_use":  row["days_since_last_use"],
            "cost_per_hour":        row["cost_per_hour"],
            "usage_trend":          row.get("usage_trend", 0),
            "days_to_renewal":      row.get("days_to_renewal", 30),
            "has_free_tier":        row.get("has_free_tier", 0),
            "has_cheaper_tier":     row.get("has_cheaper_tier", 0),
            "cheaper_saving":       row.get("cheaper_saving", 0),
            "target":               target,
            "reason":               f"user_feedback_{row['user_feedback'].lower()}",
        }
        rows.append(new_row)

    return pd.DataFrame(rows)

def retrain():
    print("=" * 50)
    print("WEEKLY RETRAIN")
    print(f"Date: {datetime.today().strftime('%Y-%m-%d')}")
    print("=" * 50)

    # ── Step 1: Load feedback ─────────────────────────────────────────────
    feedback_df = load_pending_feedback()
    if feedback_df.empty:
        print("Nothing to retrain on. Exiting.")
        return

    # ── Step 2: Convert feedback to labeled rows ──────────────────────────
    new_rows = convert_feedback_to_labels(feedback_df)
    print(f"\nNew labeled rows from feedback: {len(new_rows)}")
    print(new_rows[["user_name", "subscription_name", "target"]].to_string(index=False))

    # ── Step 3: Append to features.csv ───────────────────────────────────
    existing_features = pd.read_csv(FEATURES_FILE)
    combined = pd.concat([existing_features, new_rows], ignore_index=True)
    combined.to_csv(FEATURES_FILE, index=False)
    print(f"\nfeatures.csv updated: {len(existing_features)} → {len(combined)} rows")

    # ── Step 4: Log processed feedback ───────────────────────────────────
    feedback_df["processed_date"] = datetime.today().strftime("%Y-%m-%d")
    existing_log = pd.read_csv(FEEDBACK_LOG_FILE) if os.path.exists(FEEDBACK_LOG_FILE) else pd.DataFrame()
    log = pd.concat([existing_log, feedback_df], ignore_index=True)
    log.to_csv(FEEDBACK_LOG_FILE, index=False)

    # Remove processed rows from pending
    pending_df = pd.read_csv(PENDING_FEEDBACK_FILE)
    processed_keys = set(zip(feedback_df["user_name"], feedback_df["subscription_name"], feedback_df["prediction_date"].astype(str)))
    mask = pending_df.apply(
        lambda r: (r["user_name"], r["subscription_name"], str(r["prediction_date"])[:10]) not in processed_keys,
        axis=1
    )
    pending_df[mask].to_csv(PENDING_FEEDBACK_FILE, index=False)
    print(f"Moved {len(feedback_df)} rows from pending to feedback_log.csv")

    # ── Step 5: Retrain model ─────────────────────────────────────────────
    print("\nRetraining model...")
    cat_enc     = LabelEncoder()
    billing_enc = LabelEncoder()
    label_enc   = LabelEncoder()

    combined["category_encoded"] = cat_enc.fit_transform(combined["category"])
    combined["billing_encoded"]  = billing_enc.fit_transform(combined["billing_cycle"])
    combined["label_encoded"]    = label_enc.fit_transform(combined["target"])

    X = combined[FEATURE_COLS]
    y = combined["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

    # ── Step 6: Save updated model ────────────────────────────────────────
    joblib.dump(model,       os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(cat_enc,     os.path.join(MODEL_DIR, "cat_encoder.pkl"))
    joblib.dump(billing_enc, os.path.join(MODEL_DIR, "billing_encoder.pkl"))
    joblib.dump(label_enc,   os.path.join(MODEL_DIR, "label_encoder.pkl"))

    print(f"\n✅ Model retrained with {len(combined)} rows ({len(new_rows)} new from feedback)")
    print(f"Label distribution:\n{combined['target'].value_counts().to_string()}")

if __name__ == "__main__":
    retrain()
