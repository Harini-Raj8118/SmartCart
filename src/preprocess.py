"""Build the session-level dataset from the REAL REES46 CSVs.

  python -m src.preprocess --data-dir data/raw --sample-frac 0.02
"""
import argparse, json, os
import pandas as pd
from src.data_loader import discover_files, build_session_summary, iter_cart_session_events
from src.features import build_features, DEFAULT_WINDOW_MIN, TARGET

DEFAULT_OUT = "data/processed/session_features.csv"


def preprocess(data_dir="data/raw", files=None, sample_frac=1.0, window_minutes=DEFAULT_WINDOW_MIN,
               chunksize=2_000_000, out=DEFAULT_OUT, log=print):
    paths = discover_files(data_dir, files)
    log("Using REAL REES46 files: " + ", ".join(p.name for p in paths) + f" | sample_frac={sample_frac}")
    summary, n_events = build_session_summary(paths, sample_frac, chunksize, log)
    cart_keys = summary.loc[summary["has_cart"], "session_key"].to_numpy()
    tables = []
    for name, ev in iter_cart_session_events(paths, cart_keys, sample_frac, chunksize, log):
        # a session may span two monthly files: build once from the combined events at the end
        tables.append(ev)
    events = pd.concat(tables, ignore_index=True)
    del tables
    X = build_features(events, summary, window_minutes)
    sessions_total = len(summary)
    meta = {
        "dataset": "REES46 eCommerce behavior data (Kaggle: mkechinov/ecommerce-behavior-data-from-multi-category-store)",
        "synthetic": False, "files": [p.name for p in paths], "sample_frac": sample_frac,
        "window_minutes": window_minutes, "events_scanned": int(n_events),
        "sessions_total": int(sessions_total), "cart_sessions": int(summary["has_cart"].sum()),
        "model_sessions": int(len(X)),
        "abandoned": int(X[TARGET].sum()), "completed": int((1 - X[TARGET]).sum()),
        "abandonment_rate": float(X[TARGET].mean()),
        "session_start_min": str(X["session_start"].min()), "session_start_max": str(X["session_start"].max()),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X.to_csv(out, index=False)
    with open(out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"Saved {len(X):,} sessions -> {out}")
    return X, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build session features from REAL REES46 CSVs")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--files", nargs="*", help="specific CSV names, e.g. 2019-Oct.csv 2019-Nov.csv")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="fraction of REAL USERS to keep (not synthetic generation)")
    ap.add_argument("--window-minutes", type=float, default=DEFAULT_WINDOW_MIN,
                    help="minutes observed after the first add-to-cart before predicting")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    preprocess(a.data_dir, a.files, a.sample_frac, a.window_minutes, a.chunksize, a.out)
