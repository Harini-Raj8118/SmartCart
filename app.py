"""SmartCart - Streamlit UI (uses the model trained on REAL REES46 data).   Run:  streamlit run app.py"""
import os
import joblib
import pandas as pd
import streamlit as st

OUT = os.environ.get("SMARTCART_OUTPUTS", "outputs")
MODEL_PATH = os.path.join(OUT, "best_model.joblib")

st.set_page_config(page_title="SmartCart", page_icon="🛒", layout="wide")
st.title("🛒 SmartCart: Cart Abandonment Prediction")

if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Download the REES46 data from Kaggle into `data/raw/`, then train:\n\n"
             "`python -m src.train --data-dir data/raw --sample-frac 0.02`")
    st.stop()

bundle = joblib.load(MODEL_PATH)
meta = bundle.get("meta", {})
if meta.get("synthetic", True) or "REES46" not in meta.get("dataset", ""):
    st.error("This model was not trained on the real REES46 dataset. Re-train with `python -m src.train`.")
    st.stop()

model, features = bundle["model"], bundle["features"]
st.caption(f"Model: **{bundle['name']}** | trained on REAL REES46 data: {', '.join(meta['files'])} "
           f"(user sample {meta['sample_frac']}, {meta['model_sessions']:,} cart sessions) | "
           f"prediction point = first add-to-cart + {meta['window_minutes']} min")

with st.sidebar:
    st.header("Risk thresholds")
    hi = st.slider("HIGH if probability ≥", 0.5, 0.95, 0.75, 0.01)
    lo = st.slider("LOW if probability <", 0.05, 0.7, 0.45, 0.01)


def risk(p):
    return ("HIGH", "error") if p >= hi else ("LOW", "success") if p < lo else ("MEDIUM", "warning")


def show(p):
    level, kind = risk(p)
    st.metric("Abandonment Probability", f"{p:.1%}")
    getattr(st, kind)(f"Risk Level: **{level}**")
    st.progress(min(max(p, 0.0), 1.0))
    st.info({"HIGH": "Trigger a cart reminder, free-shipping offer or a limited-time discount.",
             "MEDIUM": "Show recommendations, reviews or a low-stock notice.",
             "LOW": "Likely to buy: avoid unnecessary discounts."}[level])


tab1, tab2, tab3 = st.tabs(["Predict a session", "Batch prediction", "Model insights"])

with tab1:
    st.write("Describe the shopping session **up to the prediction point** "
             f"(first add-to-cart + {meta['window_minutes']} min).")
    c1, c2, c3 = st.columns(3)
    views = c1.number_input("Product views", 0, 500, 6)
    adds = c1.number_input("Cart additions", 1, 100, 1)
    rems = c1.number_input("Cart removals", 0, 100, 0)
    uprod = c1.number_input("Unique products seen", 1, 500, 5)
    ucat = c1.number_input("Unique categories seen", 1, 100, 2)
    avgp = c2.number_input("Average product price", 0.0, 100000.0, 120.0)
    maxp = c2.number_input("Max product price", 0.0, 100000.0, 300.0)
    cartv = c2.number_input("Cart value (sum of added prices)", 0.0, 100000.0, 120.0)
    dur = c2.number_input("Session duration so far (sec)", 0, 20000, 240)
    ttc = c2.number_input("Seconds from session start to first cart", 0, 20000, 120)
    cats = bundle.get("main_categories") or ["unknown"]
    maincat = c3.selectbox("Main category", cats)
    hour = c3.slider("Start hour (UTC)", 0, 23, 14)
    dow = c3.selectbox("Day of week", list(range(7)), format_func=lambda d: "Mon Tue Wed Thu Fri Sat Sun".split()[d])
    prev_s = c3.number_input("Previous sessions of this user", 0, 1000, 0)
    prev_p = c3.number_input("Previous purchases of this user", 0, 1000, 0)

    n_ev = views + adds + rems
    row = {
        "n_events": n_ev, "n_views": views, "n_cart_adds": adds, "n_cart_removes": rems,
        "net_cart_additions": adds - rems, "n_unique_products": uprod, "n_unique_categories": ucat,
        "avg_price": avgp, "max_price": maxp, "cart_value": cartv, "removal_ratio": rems / adds,
        "views_to_cart_ratio": views / adds, "cart_to_view_ratio": adds / (views + 1),
        "session_duration_sec": dur, "time_to_first_cart_sec": ttc,
        "avg_time_between_actions_sec": dur / max(n_ev - 1, 1), "events_per_minute": n_ev / (dur / 60 + 1),
        "start_hour": hour, "day_of_week": dow, "is_weekend": int(dow >= 5),
        "n_prev_sessions": prev_s, "n_prev_purchases": prev_p, "is_returning_user": int(prev_s > 0),
        "main_category": maincat,
    }
    if st.button("Predict", type="primary"):
        show(float(model.predict_proba(pd.DataFrame([row])[features])[0, 1]))

with tab2:
    st.write("Upload a CSV with the columns used by the model: " + ", ".join(f"`{c}`" for c in features))
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df["abandonment_probability"] = model.predict_proba(df[features])[:, 1]
            df["risk_level"] = df["abandonment_probability"].map(lambda p: risk(p)[0])
            st.dataframe(df.sort_values("abandonment_probability", ascending=False).head(200))
            st.download_button("Download predictions", df.to_csv(index=False), "predictions.csv")

with tab3:
    summ = os.path.join(OUT, "results_summary.md")
    if os.path.exists(summ):
        st.markdown(open(summ).read())
    cols = st.columns(2)
    for i, img in enumerate(["model_comparison.png", "feature_importance.png", "roc_curves.png",
                             "pr_curves.png", "confusion_matrices.png"]):
        if os.path.exists(os.path.join(OUT, img)):
            cols[i % 2].image(os.path.join(OUT, img))
