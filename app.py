"""SmartCart - Streamlit UI for REAL REES46 model or clearly labelled demo model.

Run from the project root:
    streamlit run app.py

Real model:
    outputs/best_model.joblib

Synthetic demo model:
    outputs/demo/best_model.joblib

Demo models are only for testing the application flow; their metrics are not
valid evidence of real-world model performance.
"""
import os
import joblib
import pandas as pd
import streamlit as st

OUT = os.environ.get("SMARTCART_OUTPUTS", "outputs")
REAL_MODEL_PATH = os.path.join(OUT, "best_model.joblib")
DEMO_OUT = os.path.join(OUT, "demo")
DEMO_MODEL_PATH = os.path.join(DEMO_OUT, "best_model.joblib")

st.set_page_config(page_title="SmartCart", page_icon="🛒", layout="wide")
st.title("🛒 SmartCart: Cart Abandonment Prediction")


def load_model():
    """Prefer a verified real model; otherwise load a demo model if available."""
    if os.path.exists(REAL_MODEL_PATH):
        bundle = joblib.load(REAL_MODEL_PATH)
        meta = bundle.get("meta", {})
        if not meta.get("synthetic", True) and "REES46" in meta.get("dataset", ""):
            return bundle, meta, False

        st.warning(
            "The model in outputs/best_model.joblib is not verified as a real "
            "REES46 model. It will not be used for official predictions."
        )

    if os.path.exists(DEMO_MODEL_PATH):
        bundle = joblib.load(DEMO_MODEL_PATH)
        meta = bundle.get("meta", {})
        if meta.get("synthetic", False):
            return bundle, meta, True

        st.warning("The model in outputs/demo is not marked as synthetic; refusing to load it.")
        return None, {}, False

    return None, {}, False


bundle, meta, is_demo = load_model()

if bundle is None:
    st.error(
        "No compatible trained model found. To test the UI, run "
        "`py -m src.train --demo`. To train the real model, put REES46 CSVs "
        "in `data/raw/` and run `py -m src.train --data-dir data/raw --sample-frac 0.02`."
    )
    st.stop()

model = bundle["model"]
features = bundle["features"]
model_name = bundle.get("name", "Unknown model")

if is_demo:
    st.warning(
        "DEMO MODE — this model was trained on synthetic data. "
        "Use it only to test the application; do not report its metrics as real results."
    )
    st.caption("Model: **{}** | Synthetic demo sessions: {:,}".format(
        model_name, meta.get("model_sessions", 0)
    ))
else:
    st.success("Real REES46-trained model loaded.")
    st.caption(
        f"Model: **{model_name}** | trained on REAL REES46 data: "
        f"{', '.join(meta.get('files', []))} "
        f"(user sample {meta.get('sample_frac', 'unknown')}, "
        f"{meta.get('model_sessions', 0):,} cart sessions) | "
        f"prediction point = first add-to-cart + {meta.get('window_minutes', 'unknown')} min"
    )

with st.sidebar:
    st.header("Risk thresholds")
    hi = st.slider("HIGH if probability ≥", 0.5, 0.95, 0.75, 0.01)
    lo = st.slider("LOW if probability <", 0.05, 0.7, 0.45, 0.01)


def risk(p):
    if p >= hi:
        return "HIGH", "error"
    if p < lo:
        return "LOW", "success"
    return "MEDIUM", "warning"


def show(p):
    level, kind = risk(p)
    st.metric("Abandonment Probability", f"{p:.1%}")
    getattr(st, kind)(f"Risk Level: **{level}**")
    st.progress(min(max(float(p), 0.0), 1.0))

    if is_demo:
        st.caption("This is a synthetic-demo prediction, not a real-world probability estimate.")

    recommendations = {
        "HIGH": "Consider a cart reminder, free-shipping offer, or limited-time discount.",
        "MEDIUM": "Consider showing recommendations, reviews, or clearer product details.",
        "LOW": "The session appears lower-risk; avoid unnecessary discounts.",
    }
    st.info(recommendations[level])


tab1, tab2, tab3 = st.tabs(
    ["Predict a session", "Batch prediction", "Model insights"]
)

with tab1:
    window_minutes = meta.get("window_minutes", 5)
    st.write(
        "Describe the shopping session up to the prediction point "
        f"(first add-to-cart + {window_minutes} min)."
    )

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
    dow = c3.selectbox(
        "Day of week",
        list(range(7)),
        format_func=lambda d: "Mon Tue Wed Thu Fri Sat Sun".split()[d],
    )
    prev_s = c3.number_input("Previous sessions of this user", 0, 1000, 0)
    prev_p = c3.number_input("Previous purchases of this user", 0, 1000, 0)

    n_ev = views + adds + rems
    row = {
        "n_events": n_ev,
        "n_views": views,
        "n_cart_adds": adds,
        "n_cart_removes": rems,
        "net_cart_additions": adds - rems,
        "n_unique_products": uprod,
        "n_unique_categories": ucat,
        "avg_price": avgp,
        "max_price": maxp,
        "cart_value": cartv,
        "removal_ratio": rems / max(adds, 1),
        "views_to_cart_ratio": views / max(adds, 1),
        "cart_to_view_ratio": adds / (views + 1),
        "session_duration_sec": dur,
        "time_to_first_cart_sec": ttc,
        "avg_time_between_actions_sec": dur / max(n_ev - 1, 1),
        "events_per_minute": n_ev / (dur / 60 + 1),
        "start_hour": hour,
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "n_prev_sessions": prev_s,
        "n_prev_purchases": prev_p,
        "is_returning_user": int(prev_s > 0),
        "main_category": maincat,
    }

    if st.button("Predict", type="primary"):
        missing_row_features = [f for f in features if f not in row]
        if missing_row_features:
            st.error(
                "The manual input form does not yet provide these model features: "
                + ", ".join(missing_row_features)
            )
        else:
            input_df = pd.DataFrame([row])[features]
            probability = float(model.predict_proba(input_df)[0, 1])
            show(probability)

with tab2:
    st.write(
        "Upload a CSV with the columns used by the model: "
        + ", ".join(f"`{c}`" for c in features)
    )
    up = st.file_uploader("CSV file", type="csv")

    if up:
        df = pd.read_csv(up)
        missing = [c for c in features if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df["abandonment_probability"] = model.predict_proba(df[features])[:, 1]
            df["risk_level"] = df["abandonment_probability"].map(
                lambda p: risk(float(p))[0]
            )
            st.dataframe(
                df.sort_values("abandonment_probability", ascending=False).head(200)
            )
            st.download_button(
                "Download predictions",
                df.to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )

with tab3:
    results_dir = DEMO_OUT if is_demo else OUT
    summary_path = os.path.join(results_dir, "results_summary.md")

    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            st.markdown(f.read())

    if is_demo:
        st.info("The plots below are from synthetic data and are only for pipeline testing.")

    cols = st.columns(2)
    images = [
        "model_comparison.png",
        "feature_importance.png",
        "roc_curves.png",
        "pr_curves.png",
        "confusion_matrices.png",
    ]

    for i, img in enumerate(images):
        img_path = os.path.join(results_dir, img)
        if os.path.exists(img_path):
            cols[i % 2].image(img_path, caption=img.replace("_", " ").replace(".png", "").title())
