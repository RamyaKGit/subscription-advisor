import pandas as pd
import numpy as np
import os
import joblib
from generate_labels import compute_features, apply_rules

MODEL_DIR       = os.path.join(os.path.dirname(__file__), "model")
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
TEST_BANK_FILE  = os.path.join(DATA_DIR, "test_bank_data.csv")
TEST_USAGE_FILE = os.path.join(DATA_DIR, "test_usage_data.csv")

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

def load_model():
    model       = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    cat_enc     = joblib.load(os.path.join(MODEL_DIR, "cat_encoder.pkl"))
    billing_enc = joblib.load(os.path.join(MODEL_DIR, "billing_encoder.pkl"))
    label_enc   = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return model, cat_enc, billing_enc, label_enc

def safe_encode(encoder, values):
    known  = set(encoder.classes_)
    mapped = [v if v in known else encoder.classes_[0] for v in values]
    return encoder.transform(mapped)

def merge_with_test_data(bank_df, usage_df, user_name=None):
    """
    Combine user UI entries with the pre-populated test file data.
    Test files  → historical base (Jan 2025 → May 2026)
    UI entries  → appended on top as the most recent data
    Duplicates on (user_name, subscription_name, date) are dropped,
    keeping the UI entry (latest) over the test file entry.
    """
    merged_bank  = bank_df.copy()
    merged_usage = usage_df.copy()

    if os.path.exists(TEST_BANK_FILE):
        test_bank = pd.read_csv(TEST_BANK_FILE)
        if user_name:
            test_bank = test_bank[test_bank["user_name"] == user_name]
        merged_bank = pd.concat([test_bank, bank_df], ignore_index=True)
        merged_bank = merged_bank.drop_duplicates(
            subset=["user_name", "subscription_name", "date"], keep="last"
        )

    if os.path.exists(TEST_USAGE_FILE):
        test_usage = pd.read_csv(TEST_USAGE_FILE)
        if user_name:
            test_usage = test_usage[test_usage["user_name"] == user_name]
        merged_usage = pd.concat([test_usage, usage_df], ignore_index=True)
        merged_usage = merged_usage.drop_duplicates(
            subset=["user_name", "subscription_name", "date"], keep="last"
        )

    return merged_bank, merged_usage

def predict(bank_df, usage_df, plans_df, reference_date=None, merge_test_data=True):
    """
    Predict CLOSE / REVIEW / OPEN for each (user, subscription).

    If merge_test_data=True (default), the UI-supplied bank_df and usage_df
    are merged with test_bank_data.csv and test_usage_data.csv so the model
    always has a full historical base to work from, topped up by whatever
    the user has entered in the UI.
    """
    model, cat_enc, billing_enc, label_enc = load_model()

    if merge_test_data:
        user_name = bank_df["user_name"].iloc[0] if len(bank_df) > 0 else None
        bank_df, usage_df = merge_with_test_data(bank_df, usage_df, user_name=user_name)

    features_df = compute_features(bank_df, usage_df, plans_df, reference_date)
    if features_df.empty:
        return pd.DataFrame()

    rule_results = features_df.apply(apply_rules, axis=1)
    features_df["rule_label"] = [r[0] for r in rule_results]
    features_df["reason"]     = [r[1] for r in rule_results]

    features_df["category_encoded"] = safe_encode(cat_enc, features_df["category"])
    features_df["billing_encoded"]  = safe_encode(billing_enc, features_df["billing_cycle"])

    X          = features_df[FEATURE_COLS]
    proba      = model.predict_proba(X)
    pred_idx   = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)

    features_df["prediction"] = label_enc.inverse_transform(pred_idx)
    features_df["confidence"] = (confidence * 100).round(1)

    cols = [
        "user_name", "subscription_name", "category",
        "prediction", "confidence", "rule_label", "reason",
        "monthly_cost", "avg_hrs_per_month",
        "days_since_last_use", "cost_per_hour",
    ]
    return features_df[cols]
