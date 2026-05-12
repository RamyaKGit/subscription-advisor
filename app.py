import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, date

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
SUBS_FILE  = os.path.join(DATA_DIR, "user_subscriptions.csv")
USAGE_FILE = os.path.join(DATA_DIR, "user_usage.csv")

CATEGORIES = ["Streaming", "Security", "Productivity", "Fitness", "Utility", "Gaming"]

KNOWN_SUBS = {
    "Streaming":    ["Netflix", "Spotify", "Disney+", "Apple TV+", "YouTube Premium", "Other"],
    "Security":     ["Ring", "Nest Aware", "SimpliSafe", "ADT", "Other"],
    "Productivity": ["Notion", "Adobe CC", "Microsoft 365", "Slack", "Canva", "Other"],
    "Fitness":      ["Peloton", "MyFitnessPal", "Strava", "Nike Training", "Other"],
    "Utility":      ["Google Drive", "Dropbox", "iCloud", "LastPass", "Other"],
    "Gaming":       ["Xbox Game Pass", "PlayStation+", "Nintendo Switch Online", "EA Play", "Other"],
}

st.set_page_config(
    page_title="Subscription Advisor",
    page_icon="💳",
    layout="centered",
)

# ── File helpers (all users in one file, filtered by user_name) ───────────────

def load_all_subs():
    if os.path.exists(SUBS_FILE):
        return pd.read_csv(SUBS_FILE)
    return pd.DataFrame(columns=["user_name", "subscription_name", "category", "payment", "billing_cycle", "subscription_end_date"])

def load_user_subs(user_name):
    df = load_all_subs()
    return df[df["user_name"] == user_name].reset_index(drop=True)

def save_user_subs(user_name, user_df):
    """Replace this user's rows in the shared file, keep all other users."""
    all_df   = load_all_subs()
    others   = all_df[all_df["user_name"] != user_name]
    user_df  = user_df.copy()
    user_df["user_name"] = user_name
    combined = pd.concat([others, user_df], ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_csv(SUBS_FILE, index=False)

def load_all_usage():
    if os.path.exists(USAGE_FILE):
        return pd.read_csv(USAGE_FILE)
    return pd.DataFrame(columns=["user_name", "subscription_name", "date", "length_of_duration_watched"])

def load_user_usage(user_name):
    df = load_all_usage()
    return df[df["user_name"] == user_name].reset_index(drop=True)

def save_usage_entry(user_name, sub_name, usage_date, hours):
    """Append a single usage row to the shared usage file."""
    all_df  = load_all_usage()
    new_row = pd.DataFrame([{
        "user_name":                  user_name,
        "subscription_name":          sub_name,
        "date":                       usage_date,
        "length_of_duration_watched": hours,
    }])
    updated = pd.concat([all_df, new_row], ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    updated.to_csv(USAGE_FILE, index=False)

def clear_user_usage(user_name):
    all_df  = load_all_usage()
    cleaned = all_df[all_df["user_name"] != user_name]
    cleaned.to_csv(USAGE_FILE, index=False)

def user_exists(user_name):
    df = load_all_subs()
    return user_name in df["user_name"].values

def model_ready():
    return os.path.exists(os.path.join(MODEL_DIR, "model.pkl"))

def label_color(label):
    return {"CLOSE": "🔴", "REVIEW": "🟡", "OPEN": "🟢"}.get(label, "⚪")

# ── Session state ─────────────────────────────────────────────────────────────
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "page" not in st.session_state:
    # If user_name already set in session, keep them on page 2
    # Otherwise always start at landing page
    if st.session_state.user_name:
        st.session_state.page = 2
    else:
        st.session_state.page = 0

def go_to(page):
    st.session_state.page = page

# ── Progress indicator ────────────────────────────────────────────────────────
def show_progress(current):
    steps = ["My Subscriptions", "Log Usage", "Predictions"]
    cols  = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps), start=1):
        if i < current:
            col.markdown(f"<div style='text-align:center;color:gray;font-size:13px'>✅ {label}</div>", unsafe_allow_html=True)
        elif i == current:
            col.markdown(f"<div style='text-align:center;font-size:13px;font-weight:600'>● {label}</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;color:gray;font-size:13px'>○ {label}</div>", unsafe_allow_html=True)
    st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Landing / Name Entry
# ════════════════════════════════════════════════════════════════════════════════

if st.session_state.page == 0:
    st.title("💳 Subscription Advisor")
    st.markdown("#### Find out which subscriptions are worth keeping.")
    st.markdown("---")

    name = st.text_input("Enter your name to get started", placeholder="e.g. Alice", key="landing_name")

    if st.button("Let's go →", type="primary", use_container_width=True):
        entered_name = st.session_state.get("landing_name", "").strip()
        if not entered_name:
            st.warning("Please enter your name.")
        else:
            st.session_state.user_name = entered_name
            st.session_state.returning = user_exists(entered_name)
            st.session_state.page = 0.5  # intermediate state
            st.rerun()

if st.session_state.page == 0.5:
    user_name = st.session_state.user_name
    if st.session_state.get("returning", False):
        st.title("💳 Subscription Advisor")
        st.markdown(f"#### Welcome back, **{user_name}**!")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue →", type="primary", use_container_width=True):
                st.session_state.page = 2
                st.rerun()
        with col2:
            if st.button("Update Subscriptions", use_container_width=True):
                st.session_state.page = 1
                st.rerun()
    else:
        st.session_state.page = 1
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — My Subscriptions (new user or editing)
# ════════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 1:
    show_progress(1)
    user_name = st.session_state.user_name
    st.title("💳 My Subscriptions")
    st.caption(f"Setting up subscriptions for **{user_name}**.")

    existing = load_user_subs(user_name)

    # ── Add subscription form ─────────────────────────────────────────────
    with st.expander("➕ Add a subscription", expanded=len(existing) == 0):
        col1, col2 = st.columns(2)
        with col1:
            category    = st.selectbox("Category", CATEGORIES, key="add_cat")
            sub_options = KNOWN_SUBS.get(category, ["Other"])
            sub_choice  = st.selectbox("Subscription", sub_options, key="add_sub")
            sub_name    = st.text_input("Enter name", key="add_custom") if sub_choice == "Other" else sub_choice
        with col2:
            payment       = st.number_input("Cost ($)", min_value=0.01, value=9.99, step=0.01)
            billing_cycle = st.selectbox("Billing", ["monthly", "yearly"])
            end_date      = st.date_input("Next renewal date", value=date(2026, 12, 31))

        if st.button("Add subscription", type="primary"):
            if sub_name:
                new_row  = pd.DataFrame([{
                    "user_name":            user_name,
                    "subscription_name":    sub_name,
                    "category":             category,
                    "payment":              payment,
                    "billing_cycle":        billing_cycle,
                    "subscription_end_date":end_date.strftime("%Y-%m-%d"),
                }])
                existing = pd.concat([existing, new_row], ignore_index=True)
                save_user_subs(user_name, existing)
                st.success(f"Added {sub_name} ✓")
                st.rerun()

    # ── Subscriptions list ────────────────────────────────────────────────
    existing = load_user_subs(user_name)
    if len(existing) > 0:
        st.subheader(f"Your subscriptions ({len(existing)})")

        monthly_total = sum(
            row["payment"] if row["billing_cycle"] == "monthly" else row["payment"] / 12
            for _, row in existing.iterrows()
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total",         len(existing))
        c2.metric("Monthly spend", f"${monthly_total:.2f}")
        c3.metric("Yearly spend",  f"${monthly_total * 12:.2f}")
        st.markdown("")

        for i, row in existing.iterrows():
            c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
            c1.write(f"**{row['subscription_name']}** `{row['category']}`")
            c2.write(f"${row['payment']:.2f} / {row['billing_cycle']}")
            c3.write(f"Renews: {row['subscription_end_date']}")
            if c4.button("🗑️", key=f"del_{i}"):
                existing = existing.drop(i).reset_index(drop=True)
                save_user_subs(user_name, existing)
                st.rerun()

        st.markdown("---")
        if st.button("Continue →", type="primary", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    else:
        st.info("Add at least one subscription to continue.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Log Usage
# ════════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 2:
    show_progress(2)
    user_name = st.session_state.user_name
    st.title("📝 Log Usage")
    st.caption(f"Logging usage for **{user_name}**.")

    user_subs = load_user_subs(user_name)
    if len(user_subs) == 0:
        st.warning("No subscriptions found. Please set them up first.")
        if st.button("← Set up subscriptions"):
            st.session_state.page = 1
            st.rerun()
        st.stop()

    sub_names = user_subs["subscription_name"].tolist()

    # ── Add usage form ────────────────────────────────────────────────────
    with st.form("usage_form"):
        col1, col2 = st.columns(2)
        with col1:
            sub_name   = st.selectbox("Subscription", sub_names)
            usage_date = st.date_input("Date of use", value=date.today())
        with col2:
            hours = st.number_input(
                "Hours used",
                min_value=0.0,
                max_value=24.0,
                value=1.0,
                step=0.25,
                help="For passive apps like Ring or Google Drive, enter 0.1 to record a check-in"
            )
        submitted = st.form_submit_button("Log usage", type="primary", use_container_width=True)
        if submitted:
            save_usage_entry(user_name, sub_name, usage_date.strftime("%Y-%m-%d"), hours)
            st.success(f"Logged {hours}h of {sub_name} on {usage_date} ✓")

    # ── Usage history ─────────────────────────────────────────────────────
    usage_df = load_user_usage(user_name)
    if len(usage_df) > 0:
        st.markdown("---")
        st.subheader("Usage history")

        view_mode = st.radio("View", ["Summary", "Raw entries"], horizontal=True)
        if view_mode == "Summary":
            summary = (
                usage_df.groupby("subscription_name")["length_of_duration_watched"]
                .agg(total_hours="sum", sessions="count")
                .reset_index()
                .sort_values("total_hours", ascending=False)
            )
            summary["total_hours"] = summary["total_hours"].round(1)
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.dataframe(
                usage_df.sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True
            )

        if st.button("🗑️ Clear my usage history"):
            clear_user_usage(user_name)
            st.rerun()

    # ── Navigation ────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Edit Subscriptions", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with col2:
        if st.button("Get Predictions →", type="primary", use_container_width=True):
            st.session_state.page = 3
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predictions
# ════════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 3:
    show_progress(3)
    user_name = st.session_state.user_name
    st.title("🔮 Predictions")
    st.caption(f"Results for **{user_name}**.")

    if not model_ready():
        st.error("Model not trained yet. Run `python train_model.py` from the project folder.")
        if st.button("← Back"):
            st.session_state.page = 2
            st.rerun()
        st.stop()

    user_subs = load_user_subs(user_name)
    usage_df  = load_user_usage(user_name)
    plans_df  = pd.read_csv(os.path.join(DATA_DIR, "subscription_plan_details.csv"))

    # Build bank_df from this user's subscriptions
    bank_rows = []
    for _, row in user_subs.iterrows():
        bank_rows.append({
            "user_name":             user_name,
            "date":                  datetime.today().strftime("%Y-%m-%d"),
            "subscription_name":     row["subscription_name"],
            "payment":               row["payment"],
            "subscription_end_date": row["subscription_end_date"],
            "category":              row["category"],
            "billing_cycle":         row["billing_cycle"],
        })
    bank_df = pd.DataFrame(bank_rows)

    with st.spinner("Running predictions..."):
        try:
            from predictor import predict
            results = predict(bank_df, usage_df, plans_df)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            if st.button("← Back"):
                st.session_state.page = 2
                st.rerun()
            st.stop()

    if results.empty:
        st.warning("Not enough data to make predictions yet.")
        if st.button("← Back to Log Usage"):
            st.session_state.page = 2
            st.rerun()
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────
    close_count  = (results["prediction"] == "CLOSE").sum()
    review_count = (results["prediction"] == "REVIEW").sum()
    open_count   = (results["prediction"] == "OPEN").sum()

    saving = 0
    for _, row in results[results["prediction"] == "CLOSE"].iterrows():
        b = bank_df[bank_df["subscription_name"] == row["subscription_name"]]
        if not b.empty:
            cost = b["payment"].iloc[0]
            cyc  = b["billing_cycle"].iloc[0]
            saving += cost if cyc == "monthly" else cost / 12

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Cancel",        close_count)
    col2.metric("🟡 Review",        review_count)
    col3.metric("🟢 Keep",          open_count)
    col4.metric("💰 Monthly saving", f"${saving:.2f}")

    st.markdown("---")

    # ── Results per subscription ──────────────────────────────────────────
    st.subheader("Results")

    # Load existing feedback so buttons reflect already-given feedback
    pending_file = os.path.join(DATA_DIR, "pending_feedback.csv")
    pending_df   = pd.read_csv(pending_file) if os.path.exists(pending_file) else pd.DataFrame()

    for _, row in results.iterrows():
        icon = label_color(row["prediction"])
        sub  = row["subscription_name"]

        with st.expander(f"{icon} **{sub}** — {row['prediction']}  ({row['confidence']}% confidence)"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Category",       row["category"])
            c2.metric("Monthly cost",   f"${row['monthly_cost']:.2f}")
            c3.metric("Avg hrs/month",  f"{row['avg_hrs_per_month']:.1f}h")
            c4.metric("Days since use", int(row["days_since_last_use"]))
            st.info(f"**Reason:** {row['reason']}")
            if row["prediction"] != row["rule_label"]:
                st.caption(f"ℹ️ Rule engine suggested **{row['rule_label']}** — model overrode based on full pattern")

            st.markdown("**Was this prediction correct?**")

            # Check if feedback already given today
            today = datetime.today().strftime("%Y-%m-%d")
            existing = pending_df[
                (pending_df.get("user_name", pd.Series()) == user_name) &
                (pending_df.get("subscription_name", pd.Series()) == sub) &
                (pending_df.get("prediction_date", pd.Series()) == today)
            ] if not pending_df.empty else pd.DataFrame()

            current_feedback = existing["user_feedback"].iloc[0] if len(existing) > 0 else ""

            if current_feedback == "AGREE":
                st.success("✅ You agreed with this prediction")
            elif current_feedback == "DISAGREE":
                st.warning("❌ You disagreed with this prediction")
            else:
                fb_col1, fb_col2 = st.columns(2)
                if fb_col1.button(f"✅ Agree", key=f"agree_{sub}"):
                    if not pending_df.empty:
                        mask = (
                            (pending_df["user_name"] == user_name) &
                            (pending_df["subscription_name"] == sub) &
                            (pending_df["prediction_date"] == today)
                        )
                        pending_df.loc[mask, "user_feedback"] = "AGREE"
                        pending_df.to_csv(pending_file, index=False)
                    st.rerun()
                if fb_col2.button(f"❌ Disagree", key=f"disagree_{sub}"):
                    if not pending_df.empty:
                        mask = (
                            (pending_df["user_name"] == user_name) &
                            (pending_df["subscription_name"] == sub) &
                            (pending_df["prediction_date"] == today)
                        )
                        pending_df.loc[mask, "user_feedback"] = "DISAGREE"
                        pending_df.to_csv(pending_file, index=False)
                    st.rerun()

    # ── Navigation + download ─────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Log more usage", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    with col2:
        csv = results.to_csv(index=False)
        st.download_button(
            "⬇️ Download predictions CSV",
            data=csv,
            file_name=f"predictions_{user_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )
