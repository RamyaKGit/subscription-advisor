import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

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

def train():
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))

    # Encode categoricals
    cat_enc      = LabelEncoder()
    billing_enc  = LabelEncoder()
    label_enc    = LabelEncoder()

    df["category_encoded"] = cat_enc.fit_transform(df["category"])
    df["billing_encoded"]  = billing_enc.fit_transform(df["billing_cycle"])
    df["label_encoded"]    = label_enc.fit_transform(df["target"])

    X = df[FEATURE_COLS]
    y = df["label_encoded"]

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

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(
        y_test, y_pred,
        target_names=label_enc.classes_
    ))

    print("CONFUSION MATRIX")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=label_enc.classes_,
        columns=label_enc.classes_
    ).to_string())

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\nCross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Feature importance ────────────────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nFEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))

    # ── Save artifacts ────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,       os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(cat_enc,     os.path.join(MODEL_DIR, "cat_encoder.pkl"))
    joblib.dump(billing_enc, os.path.join(MODEL_DIR, "billing_encoder.pkl"))
    joblib.dump(label_enc,   os.path.join(MODEL_DIR, "label_encoder.pkl"))

    print("\nModel saved to model/model.pkl")
    return model, label_enc

if __name__ == "__main__":
    train()
