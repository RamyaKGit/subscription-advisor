import pandas as pd
import numpy as np
from datetime import datetime, date
import random
import os

random.seed(99)
np.random.seed(99)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Test user's subscriptions ─────────────────────────────────────────────────
# We define a realistic mix of personas so predictions are interesting
# (not all OPEN, not all CLOSE)

TEST_USER = "test_user"

TEST_SUBS = [
    # Should → OPEN  (heavy streamer)
    {"subscription_name": "Netflix",        "category": "Streaming",    "payment": 17.99, "billing_cycle": "monthly", "persona": "heavy"},
    # Should → OPEN  (moderate streamer)
    {"subscription_name": "Spotify",        "category": "Streaming",    "payment": 10.99, "billing_cycle": "monthly", "persona": "moderate"},
    # Should → CLOSE (barely watches)
    {"subscription_name": "Disney+",        "category": "Streaming",    "payment": 13.99, "billing_cycle": "monthly", "persona": "inactive"},
    # Should → REVIEW (seasonal - only used in certain months)
    {"subscription_name": "Peloton",        "category": "Fitness",      "payment": 44.00, "billing_cycle": "monthly", "persona": "seasonal"},
    # Should → OPEN  (used regularly for work)
    {"subscription_name": "Notion",         "category": "Productivity", "payment": 10.00, "billing_cycle": "monthly", "persona": "moderate"},
    # Should → CLOSE (paid tier but barely uses it)
    {"subscription_name": "Adobe CC",       "category": "Productivity", "payment": 59.99, "billing_cycle": "yearly",  "persona": "inactive"},
    # Should → OPEN  (passive security, logs in regularly)
    {"subscription_name": "Ring",           "category": "Security",     "payment": 10.00, "billing_cycle": "monthly", "persona": "moderate"},
    # Should → CLOSE (never opened, free tier would cover it)
    {"subscription_name": "Dropbox",        "category": "Utility",      "payment": 11.99, "billing_cycle": "monthly", "persona": "inactive"},
    # Should → REVIEW (binge gamed early in year, dropped off)
    {"subscription_name": "Xbox Game Pass", "category": "Gaming",       "payment": 14.99, "billing_cycle": "monthly", "persona": "binge_drop"},
    # Should → OPEN  (consistent gym tracker)
    {"subscription_name": "MyFitnessPal",   "category": "Fitness",      "payment": 9.99,  "billing_cycle": "monthly", "persona": "heavy"},
]

# ── Date range: Jan 2025 → May 2026 ──────────────────────────────────────────
START = datetime(2025, 1, 1)
END   = datetime(2026, 5, 31)

def months_between(start, end):
    months = []
    cur = start.replace(day=1)
    while cur <= end:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year+1, month=1)
        else:
            cur = cur.replace(month=cur.month+1)
    return months

def add_noise(value, pct=0.10):
    return max(0, value + np.random.normal(0, pct * abs(value)))

def hours_for_persona(category, persona, month):
    """Return realistic hours for the given category + persona + month."""
    base = {
        "Streaming":    28,
        "Security":      2,
        "Productivity": 14,
        "Fitness":       8,
        "Utility":       1,
        "Gaming":       22,
    }.get(category, 10)

    if persona == "heavy":
        mult = random.uniform(1.6, 2.4)
    elif persona == "moderate":
        mult = random.uniform(0.8, 1.4)
    elif persona == "inactive":
        # Very occasionally opens it
        mult = random.uniform(0.0, 0.08) if random.random() > 0.15 else random.uniform(0.1, 0.3)
    elif persona == "seasonal":
        # Active Jan, Apr, Sep, Dec, Nov — quiet otherwise
        if month in [1, 4, 9, 11, 12]:
            mult = random.uniform(1.4, 2.2)
        else:
            mult = random.uniform(0.0, 0.05)
    elif persona == "binge_drop":
        # 2025: heavy Jan-Mar, near zero Apr onwards; 2026: still zero
        if month <= 3:
            mult = random.uniform(2.0, 3.0)
        else:
            mult = random.uniform(0.0, 0.04)
    else:
        mult = 1.0

    return round(add_noise(base * mult, pct=0.12), 2)

def generate_sessions(category, persona, year, month, total_hours):
    """Break total hours into individual session rows."""
    if total_hours <= 0:
        return []

    avg_session = {
        "Streaming":    1.5,
        "Security":     0.1,
        "Productivity": 1.0,
        "Fitness":      0.75,
        "Utility":      0.05,
        "Gaming":       2.0,
    }.get(category, 1.0)

    num_sessions = max(1, int(total_hours / avg_session))
    days_in_month = 28 if month == 2 else 30 if month in [4,6,9,11] else 31
    session_days  = sorted(random.sample(
        range(1, days_in_month + 1),
        min(num_sessions, days_in_month)
    ))

    sessions = []
    remaining = total_hours
    for i, day in enumerate(session_days):
        if i == len(session_days) - 1:
            dur = round(max(0.05, remaining), 2)
        else:
            dur = round(random.uniform(0.1, avg_session * 1.5), 2)
            remaining = max(0, remaining - dur)
        sessions.append({"day": day, "duration": round(add_noise(dur, 0.08), 2)})

    return sessions

# ── Build the dataframes ──────────────────────────────────────────────────────

def generate_test_files():
    months     = months_between(START, END)
    bank_rows  = []
    usage_rows = []

    for sub in TEST_SUBS:
        name     = sub["subscription_name"]
        cat      = sub["category"]
        payment  = sub["payment"]
        billing  = sub["billing_cycle"]
        persona  = sub["persona"]

        # End date: one year from Jan 2025 for yearly, rolling monthly for monthly
        if billing == "yearly":
            sub_end = "2026-01-01"
        else:
            sub_end = "2026-06-01"

        for (year, month) in months:
            # ── Bank entry ────────────────────────────────────────────────
            if billing == "monthly":
                pay_date    = datetime(year, month, random.randint(1, 5))
                noisy_price = round(add_noise(payment, 0.02), 2)
                end_date    = "2026-06-01"
                bank_rows.append({
                    "user_name":             TEST_USER,
                    "date":                  pay_date.strftime("%Y-%m-%d"),
                    "subscription_name":     name,
                    "payment":               noisy_price,
                    "subscription_end_date": end_date,
                    "category":              cat,
                    "billing_cycle":         "monthly",
                })
            elif billing == "yearly":
                # One payment per year (Jan)
                if month == 1:
                    yearly_price = round(add_noise(payment * 12 * 0.85, 0.02), 2)
                    bank_rows.append({
                        "user_name":             TEST_USER,
                        "date":                  datetime(year, 1, 2).strftime("%Y-%m-%d"),
                        "subscription_name":     name,
                        "payment":               yearly_price,
                        "subscription_end_date": sub_end,
                        "category":              cat,
                        "billing_cycle":         "yearly",
                    })

            # ── Usage entries ─────────────────────────────────────────────
            total_hours = hours_for_persona(cat, persona, month)
            sessions    = generate_sessions(cat, persona, year, month, total_hours)
            days_in_month = 28 if month == 2 else 30 if month in [4,6,9,11] else 31

            for s in sessions:
                day = min(s["day"], days_in_month)
                usage_rows.append({
                    "user_name":                  TEST_USER,
                    "subscription_name":          name,
                    "date":                       datetime(year, month, day).strftime("%Y-%m-%d"),
                    "length_of_duration_watched": s["duration"],
                })

    bank_df  = pd.DataFrame(bank_rows)
    usage_df = pd.DataFrame(usage_rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bank_df.to_csv(os.path.join(OUTPUT_DIR,  "test_bank_data.csv"),  index=False)
    usage_df.to_csv(os.path.join(OUTPUT_DIR, "test_usage_data.csv"), index=False)

    print(f"test_bank_data.csv   → {len(bank_df)} rows")
    print(f"test_usage_data.csv  → {len(usage_df)} rows")
    print()
    print("Subscriptions in test data:")
    for sub in TEST_SUBS:
        print(f"  {sub['subscription_name']:<20} persona={sub['persona']:<12} → expected: ", end="")
        p = sub["persona"]
        cat = sub["category"]
        if p == "inactive":
            print("CLOSE")
        elif p == "binge_drop":
            print("REVIEW / CLOSE")
        elif p == "seasonal":
            print("REVIEW")
        elif p in ["heavy", "moderate"]:
            print("OPEN")

    return bank_df, usage_df

if __name__ == "__main__":
    generate_test_files()
