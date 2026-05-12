import pandas as pd
import numpy as np
import os
import random

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Category-aware thresholds ─────────────────────────────────────────────────

THRESHOLDS = {
    "Streaming": {
        # Based on hours per month average
        "close_hrs":  10,
        "review_hrs": 20,
    },
    "Security": {
        # Based on days since last login
        "close_days":  90,
        "review_days": 60,
    },
    "Productivity": {
        # Based on number of active months in last 3 months
        "close_active_months":  0,
        "review_active_months": 1,
    },
    "Fitness": {
        # Based on total sessions in last 3 months
        "close_sessions":  4,
        "review_sessions": 8,
    },
    "Utility": {
        # Based on whether free/cheaper tier exists and covers usage
        # Handled separately via plan lookup
    },
    "Gaming": {
        # Based on active months in last 3 months
        "close_active_months":  0,
        "review_active_months": 1,
    },
}

def compute_features(bank_df, usage_df, plans_df, reference_date=None):
    """
    For each (user, subscription) pair compute all features needed
    to apply rules and train the model.
    """
    if reference_date is None:
        reference_date = pd.Timestamp("2025-12-31")

    plan_lookup = plans_df.set_index("subscription_name").to_dict("index")

    # Ensure date columns are datetime
    bank_df  = bank_df.copy()
    usage_df = usage_df.copy()
    bank_df["date"]  = pd.to_datetime(bank_df["date"])
    usage_df["date"] = pd.to_datetime(usage_df["date"])
    bank_df["subscription_end_date"] = pd.to_datetime(bank_df["subscription_end_date"])

    # Last 3 months window
    three_months_ago = reference_date - pd.DateOffset(months=3)

    records = []

    for (user, sub_name), u_group in usage_df.groupby(["user_name", "subscription_name"]):
        b_group = bank_df[
            (bank_df["user_name"] == user) &
            (bank_df["subscription_name"] == sub_name)
        ]
        if b_group.empty:
            continue

        category     = b_group["category"].iloc[0]
        billing      = b_group["billing_cycle"].iloc[0]
        plan         = plan_lookup.get(sub_name, {})

        # ── Cost features ──────────────────────────────────────────────────
        if billing == "monthly":
            monthly_cost = b_group["payment"].mean()
        else:
            monthly_cost = b_group["payment"].sum() / 12.0

        # ── Usage features ─────────────────────────────────────────────────
        total_hours      = u_group["length_of_duration_watched"].sum()
        total_sessions   = len(u_group)

        # Last 3 months usage
        recent = u_group[u_group["date"] >= three_months_ago]
        recent_hours     = recent["length_of_duration_watched"].sum()
        recent_sessions  = len(recent)

        # Active months in last 3 months
        recent_months = recent["date"].dt.to_period("M").nunique()

        # Days since last use
        if len(u_group) > 0:
            last_use_date    = u_group["date"].max()
            days_since_use   = (reference_date - last_use_date).days
        else:
            days_since_use   = 999

        # Average hours per month (over 12 months)
        avg_hrs_per_month = total_hours / 12.0

        # Cost per hour (avoid div/0)
        cost_per_hour = monthly_cost / avg_hrs_per_month if avg_hrs_per_month > 0 else 999

        # Usage trend: compare last 3 months avg vs previous 3 months avg
        six_months_ago = reference_date - pd.DateOffset(months=6)
        prev_period = u_group[
            (u_group["date"] >= six_months_ago) &
            (u_group["date"] < three_months_ago)
        ]
        prev_hrs   = prev_period["length_of_duration_watched"].sum()
        prev_avg   = prev_hrs / 3.0
        recent_avg = recent_hours / 3.0
        if prev_avg > 0:
            usage_trend = (recent_avg - prev_avg) / prev_avg
        else:
            usage_trend = 0.0

        # Days to renewal
        latest_end = b_group["subscription_end_date"].max()
        days_to_renewal = (latest_end - reference_date).days

        # Plan features
        has_free_tier     = 1 if plan.get("free_tier") else 0
        has_cheaper_tier  = 1 if plan.get("medium_price") and plan.get("medium_price", 999) < monthly_cost else 0
        free_price        = plan.get("free_price") or 0
        medium_price      = plan.get("medium_price") or monthly_cost
        cheaper_saving    = monthly_cost - medium_price

        records.append({
            "user_name":           user,
            "subscription_name":   sub_name,
            "category":            category,
            "billing_cycle":       billing,
            "monthly_cost":        round(monthly_cost, 2),
            "total_hours":         round(total_hours, 2),
            "total_sessions":      total_sessions,
            "avg_hrs_per_month":   round(avg_hrs_per_month, 2),
            "recent_hours":        round(recent_hours, 2),
            "recent_sessions":     recent_sessions,
            "recent_active_months":recent_months,
            "days_since_last_use": days_since_use,
            "cost_per_hour":       round(cost_per_hour, 2),
            "usage_trend":         round(usage_trend, 4),
            "days_to_renewal":     days_to_renewal,
            "has_free_tier":       has_free_tier,
            "has_cheaper_tier":    has_cheaper_tier,
            "cheaper_saving":      round(cheaper_saving, 2),
        })

    return pd.DataFrame(records)

# ── Category-aware rule engine ────────────────────────────────────────────────

def apply_rules(row):
    cat = row["category"]
    t   = THRESHOLDS.get(cat, {})

    if cat == "Streaming":
        avg = row["avg_hrs_per_month"]
        if avg < t["close_hrs"]:
            return "CLOSE", "Average hours per month below threshold"
        elif avg < t["review_hrs"]:
            return "REVIEW", "Moderate usage — consider downgrade"
        else:
            return "OPEN", "Good usage"

    elif cat == "Security":
        days = row["days_since_last_use"]
        if days >= t["close_days"]:
            return "CLOSE", f"No login in {days} days"
        elif days >= t["review_days"]:
            return "REVIEW", f"Low login frequency ({days} days since last use)"
        else:
            return "OPEN", "Regular logins detected"

    elif cat in ["Productivity", "Gaming"]:
        active = row["recent_active_months"]
        if active <= t["close_active_months"]:
            return "CLOSE", "Zero active months in last 3 months"
        elif active <= t["review_active_months"]:
            return "REVIEW", "Used in only 1 of last 3 months"
        else:
            return "OPEN", "Active in 2+ of last 3 months"

    elif cat == "Fitness":
        sessions = row["recent_sessions"]
        if sessions < t["close_sessions"]:
            return "CLOSE", f"Only {sessions} sessions in last 3 months"
        elif sessions < t["review_sessions"]:
            return "REVIEW", f"Moderate activity ({sessions} sessions)"
        else:
            return "OPEN", "Good activity level"

    elif cat == "Utility":
        if row["has_free_tier"] and row["avg_hrs_per_month"] < 2:
            return "CLOSE", "Free tier available and usage is minimal"
        elif row["has_cheaper_tier"] and row["cheaper_saving"] > 2:
            return "REVIEW", f"Cheaper tier available — save £{row['cheaper_saving']:.2f}/mo"
        else:
            return "OPEN", "On appropriate plan"

    # Fallback
    if row["days_since_last_use"] > 90:
        return "CLOSE", "No usage in 90+ days"
    return "OPEN", "No issues detected"

def add_noise(df, flip_rate=0.12):
    """
    Flip ~12% of labels to adjacent class only.
    CLOSE ↔ REVIEW ↔ OPEN  (never CLOSE → OPEN directly)
    """
    label_order = ["CLOSE", "REVIEW", "OPEN"]
    flipped = df["target"].tolist()
    reasons = df["reason"].tolist()

    flip_indices = random.sample(range(len(flipped)), int(len(flipped) * flip_rate))
    for i in flip_indices:
        current_idx = label_order.index(flipped[i])
        direction   = random.choice([-1, 1])
        new_idx     = max(0, min(2, current_idx + direction))
        flipped[i]  = label_order[new_idx]
        reasons[i]  = reasons[i] + " [noise]"

    df = df.copy()
    df["target"] = flipped
    df["reason"] = reasons
    return df

def generate_labels(add_label_noise=True):
    bank_df  = pd.read_csv(os.path.join(DATA_DIR, "bank_data.csv"))
    usage_df = pd.read_csv(os.path.join(DATA_DIR, "usage_data.csv"))
    plans_df = pd.read_csv(os.path.join(DATA_DIR, "subscription_plan_details.csv"))

    features_df = compute_features(bank_df, usage_df, plans_df)

    results = features_df.apply(apply_rules, axis=1)
    features_df["target"] = [r[0] for r in results]
    features_df["reason"] = [r[1] for r in results]

    if add_label_noise:
        features_df = add_noise(features_df, flip_rate=0.12)

    suggestion_sheet = features_df[["user_name", "subscription_name", "category", "target", "reason"]]
    suggestion_sheet.to_csv(os.path.join(DATA_DIR, "suggestion_sheet.csv"), index=False)

    # Save full feature set for model training
    features_df.to_csv(os.path.join(DATA_DIR, "features.csv"), index=False)

    print(f"suggestion_sheet.csv → {len(suggestion_sheet)} rows")
    print(f"Label distribution:\n{features_df['target'].value_counts().to_string()}")

    return features_df

if __name__ == "__main__":
    generate_labels()
