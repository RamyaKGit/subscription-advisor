import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
START_DATE = datetime(2025, 1, 1)
END_DATE   = datetime(2025, 12, 31)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

USERS = [f"user_{i:02d}" for i in range(1, 51)]   # 50 users

# ── Subscription plan details ─────────────────────────────────────────────────
PLANS = [
    # Streaming
    {"subscription_name": "Netflix",       "category": "Streaming",    "free_tier": None,  "free_price": None, "medium_tier": "Standard", "medium_price": 10.99, "costly_tier": "Premium",  "costly_price": 17.99},
    {"subscription_name": "Spotify",       "category": "Streaming",    "free_tier": "Free","free_price": 0.00, "medium_tier": "Individual","medium_price": 10.99, "costly_tier": "Family",   "costly_price": 17.99},
    {"subscription_name": "Disney+",       "category": "Streaming",    "free_tier": None,  "free_price": None, "medium_tier": "Standard", "medium_price": 7.99,  "costly_tier": "Premium",  "costly_price": 13.99},
    # Security / Home
    {"subscription_name": "Ring",          "category": "Security",     "free_tier": "Basic","free_price": 0.00,"medium_tier": "Plus",    "medium_price": 10.00, "costly_tier": "Pro",      "costly_price": 20.00},
    {"subscription_name": "Nest Aware",    "category": "Security",     "free_tier": None,  "free_price": None, "medium_tier": "Standard", "medium_price": 6.00,  "costly_tier": "Premium",  "costly_price": 12.00},
    # Productivity
    {"subscription_name": "Notion",        "category": "Productivity", "free_tier": "Free","free_price": 0.00, "medium_tier": "Plus",    "medium_price": 10.00, "costly_tier": "Business", "costly_price": 18.00},
    {"subscription_name": "Adobe CC",     "category": "Productivity", "free_tier": None,  "free_price": None, "medium_tier": "Single App","medium_price":29.99, "costly_tier": "All Apps", "costly_price": 59.99},
    # Fitness
    {"subscription_name": "Peloton",       "category": "Fitness",      "free_tier": None,  "free_price": None, "medium_tier": "Digital",  "medium_price": 12.99, "costly_tier": "All-Access","costly_price": 44.00},
    {"subscription_name": "MyFitnessPal",  "category": "Fitness",      "free_tier": "Free","free_price": 0.00, "medium_tier": "Premium",  "medium_price": 9.99,  "costly_tier": None,       "costly_price": None},
    # Utility
    {"subscription_name": "Google Drive",  "category": "Utility",      "free_tier": "Free","free_price": 0.00, "medium_tier": "100GB",   "medium_price": 2.99,  "costly_tier": "2TB",      "costly_price": 9.99},
    {"subscription_name": "Dropbox",       "category": "Utility",      "free_tier": "Free","free_price": 0.00, "medium_tier": "Plus",    "medium_price": 11.99, "costly_tier": "Professional","costly_price":19.99},
    # Gaming
    {"subscription_name": "Xbox Game Pass","category": "Gaming",       "free_tier": None,  "free_price": None, "medium_tier": "Core",    "medium_price": 7.99,  "costly_tier": "Ultimate", "costly_price": 14.99},
    {"subscription_name": "PlayStation+",  "category": "Gaming",       "free_tier": None,  "free_price": None, "medium_tier": "Essential","medium_price": 8.99, "costly_tier": "Premium",  "costly_price": 17.99},
]

PLAN_LOOKUP = {p["subscription_name"]: p for p in PLANS}

# ── User personas ─────────────────────────────────────────────────────────────
# Each user gets a persona per subscription that drives their usage pattern
PERSONAS = ["heavy", "moderate", "light", "seasonal", "inactive", "binge_drop"]

PERSONA_WEIGHTS = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]

def assign_tier(plan):
    """Randomly assign a user to a paid tier (medium or costly)."""
    if plan["costly_tier"]:
        return random.choice(["medium", "costly"])
    return "medium"

def get_price(plan, tier):
    if tier == "medium":
        return plan["medium_price"]
    return plan["costly_price"]

def date_range_months(start, end):
    """Return list of (year, month) tuples between start and end."""
    months = []
    cur = start.replace(day=1)
    while cur <= end:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months

def add_feature_noise(value, pct=0.12):
    """Add ±12% Gaussian noise to a numeric value."""
    noise = np.random.normal(0, pct * abs(value))
    return max(0, value + noise)

# ── Usage generators per category & persona ──────────────────────────────────

def generate_usage_hours(category, persona, month):
    """Return hours used in a given month based on category + persona."""

    # Base hours by category
    base = {
        "Streaming":   25,
        "Security":     2,   # logins/checks, not watch time
        "Productivity": 15,
        "Fitness":      8,
        "Utility":      1,
        "Gaming":       20,
    }.get(category, 10)

    multiplier = 1.0

    if persona == "heavy":
        multiplier = random.uniform(1.5, 2.5)
    elif persona == "moderate":
        multiplier = random.uniform(0.8, 1.4)
    elif persona == "light":
        multiplier = random.uniform(0.2, 0.6)
    elif persona == "inactive":
        multiplier = random.uniform(0.0, 0.05)
    elif persona == "seasonal":
        # High in Jan, Apr, Sep, Dec — low otherwise
        if month in [1, 4, 9, 12]:
            multiplier = random.uniform(1.5, 2.5)
        else:
            multiplier = random.uniform(0.0, 0.1)
    elif persona == "binge_drop":
        # Heavy first 3 months, near zero after
        if month <= 3:
            multiplier = random.uniform(2.0, 3.0)
        else:
            multiplier = random.uniform(0.0, 0.08)

    hours = base * multiplier
    hours = add_feature_noise(hours, pct=0.12)
    return round(max(0, hours), 2)

def generate_sessions(category, persona, month, hours):
    """Generate individual session rows for a month."""
    if hours == 0:
        return []

    # Average session length by category
    avg_session = {
        "Streaming":   1.5,
        "Security":    0.1,
        "Productivity":1.0,
        "Fitness":     0.75,
        "Utility":     0.05,
        "Gaming":      2.0,
    }.get(category, 1.0)

    num_sessions = max(1, int(hours / avg_session))
    sessions = []

    # Spread sessions across the month
    days_in_month = 28 if month == 2 else 30 if month in [4,6,9,11] else 31
    session_days = sorted(random.sample(range(1, days_in_month + 1), min(num_sessions, days_in_month)))

    remaining = hours
    for i, day in enumerate(session_days):
        if i == len(session_days) - 1:
            duration = round(max(0.05, remaining), 2)
        else:
            duration = round(random.uniform(0.1, avg_session * 1.5), 2)
            remaining = max(0, remaining - duration)
        sessions.append({
            "day": day,
            "duration": add_feature_noise(duration, pct=0.1)
        })

    return sessions

# ── Main generation ───────────────────────────────────────────────────────────

def generate_all():
    bank_rows    = []
    usage_rows   = []
    months       = date_range_months(START_DATE, END_DATE)

    # Each user subscribes to a random subset of 4–7 subscriptions
    user_subs = {}
    for user in USERS:
        num_subs = random.randint(4, 7)
        chosen   = random.sample(PLANS, num_subs)
        subs     = []
        for plan in chosen:
            tier    = assign_tier(plan)
            price   = get_price(plan, tier)
            persona = random.choices(PERSONAS, weights=PERSONA_WEIGHTS)[0]
            billing = random.choice(["monthly", "yearly"])
            subs.append({
                "plan": plan,
                "tier": tier,
                "price": price,
                "persona": persona,
                "billing": billing,
            })
        user_subs[user] = subs

    for user, subs in user_subs.items():
        for sub in subs:
            plan    = sub["plan"]
            price   = sub["price"]
            billing = sub["billing"]
            persona = sub["persona"]
            name    = plan["subscription_name"]
            cat     = plan["category"]

            for (year, month) in months:
                # Bank entry — monthly billing every month, yearly billing once
                if billing == "monthly":
                    pay_date = datetime(year, month, random.randint(1, 5))
                    noisy_price = round(add_feature_noise(price, pct=0.02), 2)
                    end_date = (pay_date + timedelta(days=30)).strftime("%Y-%m-%d")
                    bank_rows.append({
                        "user_name":            user,
                        "date":                 pay_date.strftime("%Y-%m-%d"),
                        "subscription_name":    name,
                        "payment":              noisy_price,
                        "subscription_end_date":end_date,
                        "category":             cat,
                        "billing_cycle":        "monthly",
                    })
                elif billing == "yearly" and month == 1:
                    pay_date  = datetime(year, 1, random.randint(1, 5))
                    yearly_price = round(add_feature_noise(price * 12 * 0.85, pct=0.02), 2)
                    end_date  = datetime(year, 12, 31).strftime("%Y-%m-%d")
                    bank_rows.append({
                        "user_name":            user,
                        "date":                 pay_date.strftime("%Y-%m-%d"),
                        "subscription_name":    name,
                        "payment":              yearly_price,
                        "subscription_end_date":end_date,
                        "category":             cat,
                        "billing_cycle":        "yearly",
                    })

                # Usage entries
                hours = generate_usage_hours(cat, persona, month)
                sessions = generate_sessions(cat, persona, month, hours)
                days_in_month = 28 if month == 2 else 30 if month in [4,6,9,11] else 31
                for s in sessions:
                    day = min(s["day"], days_in_month)
                    usage_rows.append({
                        "user_name":                user,
                        "subscription_name":        name,
                        "date":                     datetime(year, month, day).strftime("%Y-%m-%d"),
                        "length_of_duration_watched": round(s["duration"], 2),
                    })

    bank_df  = pd.DataFrame(bank_rows)
    usage_df = pd.DataFrame(usage_rows)
    plans_df = pd.DataFrame(PLANS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bank_df.to_csv(os.path.join(OUTPUT_DIR,  "bank_data.csv"),                 index=False)
    usage_df.to_csv(os.path.join(OUTPUT_DIR, "usage_data.csv"),                index=False)
    plans_df.to_csv(os.path.join(OUTPUT_DIR, "subscription_plan_details.csv"), index=False)

    print(f"bank_data.csv        → {len(bank_df)} rows")
    print(f"usage_data.csv       → {len(usage_df)} rows")
    print(f"subscription_plan_details.csv → {len(plans_df)} rows")
    return bank_df, usage_df, plans_df

if __name__ == "__main__":
    generate_all()
