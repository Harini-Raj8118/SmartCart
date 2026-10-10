"""Train SmartCart models on real REES46 data or generated demo sessions.

Real data:
  python -m src.train --data-dir data/raw --sample-frac 0.02
  python -m src.train --features data/processed/session_features.csv

Demo only (does not require the large REES46 download):
  python -m src.train --demo
  python -m src.train --demo --demo-rows 1500

Demo metrics are for pipeline smoke-testing only. They are NOT research results.
"""
import argparse, json, os, time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import CATEGORICAL, NUMERIC, TARGET
from src.preprocess import DEFAULT_OUT, preprocess

OUT = "outputs"
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def make_demo_sessions(n_rows=1000, seed=42):
    """Create session-level synthetic rows matching src.features; demo use only."""
    if n_rows < 200:
        raise ValueError("Use at least 200 demo rows so the train/test split and CV can run.")

    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], int(np.ceil(n_rows / 2)))[:n_rows]
    rng.shuffle(y)
    abandoned = y == 1

    n_cart_adds = rng.integers(1, 5, size=n_rows)
    n_cart_removes = np.where(
        abandoned,
        rng.integers(1, 4, size=n_rows),
        rng.integers(0, 2, size=n_rows),
    )
    n_views = rng.poisson(np.where(abandoned, 5.0, 3.5))
    n_events = n_views + n_cart_adds + n_cart_removes
    n_unique_products = np.maximum(1, np.minimum(n_events, rng.poisson(2.5, n_rows) + 1))
    n_unique_categories = np.maximum(1, np.minimum(n_unique_products, rng.integers(1, 4, n_rows)))
    avg_price = rng.lognormal(mean=3.2, sigma=0.7, size=n_rows).clip(2, 500)
    max_price = avg_price * rng.uniform(1.0, 2.5, size=n_rows)
    cart_value = n_cart_adds * avg_price * rng.uniform(0.7, 1.3, size=n_rows)
    session_duration = rng.gamma(shape=2.0, scale=np.where(abandoned, 100.0, 75.0))
    time_to_cart = rng.uniform(0, 180, size=n_rows)
    day_of_week = rng.integers(0, 7, size=n_rows)
    start_hour = rng.integers(0, 24, size=n_rows)
    n_prev_sessions = rng.poisson(2.0, size=n_rows)
    n_prev_purchases = np.minimum(n_prev_sessions, rng.poisson(0.8, size=n_rows))

    # Synthetic patterns are intentionally varied to exercise preprocessing and model code.
    X = pd.DataFrame({
        "session_key": np.arange(1, n_rows + 1),
        "user_id": np.arange(100_001, 100_001 + n_rows),
        "session_start": pd.date_range("2025-01-01", periods=n_rows, freq="h", tz="UTC"),
        "n_events": n_events,
        "n_views": n_views,
        "n_cart_adds": n_cart_adds,
        "n_cart_removes": n_cart_removes,
        "net_cart_additions": n_cart_adds - n_cart_removes,
        "n_unique_products": n_unique_products,
        "n_unique_categories": n_unique_categories,
        "avg_price": avg_price,
        "max_price": max_price,
        "cart_value": cart_value,
        "removal_ratio": n_cart_removes / n_cart_adds,
        "views_to_cart_ratio": n_views / n_cart_adds,
        "cart_to_view_ratio": n_cart_adds / (n_views + 1),
        "session_duration_sec": session_duration,
        "time_to_first_cart_sec": time_to_cart,
        "avg_time_between_actions_sec": session_duration / np.maximum(n_events - 1, 1),
        "events_per_minute": n_events / (session_duration / 60 + 1),
        "start_hour": start_hour,
        "day_of_week": day_of_week,
        "is_weekend": (day_of_week >= 5).astype(int),
        "n_prev_sessions": n_prev_sessions,
        "n_prev_purchases": n_prev_purchases,
        "is_returning_user": (n_prev_sessions > 0).astype(int),
        "main_category": rng.choice(
            ["electronics", "computers", "apparel", "home", "sport", "unknown"],
            size=n_rows,
        ),
        TARGET: y.astype(int),
    })

    return X


def load_sessions(a):
    # Demo mode is isolated from the real-data path and never labels itself as REES46.
    if getattr(a, "demo", False):
        if a.features:
            raise ValueError("Use either --demo or --features, not both.")
        X = make_demo_sessions(a.demo_rows, a.seed)
        meta = {
            "dataset": "Synthetic SmartCart demo sessions (generated; not REES46)",
            "synthetic": True,
            "files": ["generated_in_memory"],
            "sample_frac": 1.0,
            "window_minutes": a.window_minutes,
            "events_scanned": 0,
            "sessions_total": int(len(X)),
            "cart_sessions": int(len(X)),
            "model_sessions": int(len(X)),
            "abandoned": int(X[TARGET].sum()),
            "completed": int((1 - X[TARGET]).sum()),
            "abandonment_rate": float(X[TARGET].mean()),
            "session_start_min": str(X["session_start"].min()),
            "session_start_max": str(X["session_start"].max()),
        }
        return X, meta

    if a.features:
        if not os.path.exists(a.features) or not os.path.exists(a.features + ".meta.json"):
            raise FileNotFoundError(
                "Preprocessed table or its .meta.json is missing. Re-run src.preprocess "
                "on the real REES46 files (unknown/synthetic tables are blocked)."
            )
        X = pd.read_csv(a.features, parse_dates=["session_start"])
        with open(a.features + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        X, meta = preprocess(
            a.data_dir, a.files, a.sample_frac, a.window_minutes,
            a.chunksize, DEFAULT_OUT
        )

    if meta.get("synthetic", True) or "REES46" not in meta.get("dataset", ""):
        raise ValueError("Refusing to train: data is not marked as real REES46 data.")

    required = ["session_start", "user_id", TARGET, *NUMERIC, *CATEGORICAL]
    missing = sorted(set(required) - set(X.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")
    if X.empty:
        raise ValueError("Feature dataset is empty.")
    if X[TARGET].nunique() != 2:
        raise ValueError("Training requires both abandoned and completed sessions.")
    return X, meta


def time_split(X, test_size):
    """Earlier sessions -> train, later sessions -> test (no random mixing, no look-ahead)."""
    X = X.sort_values("session_start").reset_index(drop=True)
    cut = int(len(X) * (1 - test_size))
    return X.iloc[:cut], X.iloc[cut:]


def select_features(Xtr, thr=0.95):
    num = [c for c in NUMERIC if Xtr[c].nunique() > 1]                     # drop constant columns
    corr = Xtr[num].corr().abs()
    up = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped_corr = [c for c in num if (up[c] > thr).any()]
    num = [c for c in num if c not in dropped_corr]
    cat = [c for c in CATEGORICAL if Xtr[c].nunique() > 1]
    return num, cat, dropped_corr


def preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001), cat)])


def score(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba),
            "pr_auc": average_precision_score(y, proba)}, pred


def md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def main(a):
    global OUT
    demo_mode = getattr(a, "demo", False)
    if demo_mode:
        OUT = "outputs/demo"
    t0 = time.time()
    os.makedirs(f"{OUT}/powerbi", exist_ok=True)
    sessions, meta = load_sessions(a)
    n, n_ab = len(sessions), int(sessions[TARGET].sum())
    data_label = "SYNTHETIC DEMO" if demo_mode else "REAL REES46"
    print(f"\n{data_label} sessions: {n:,} | abandoned: {n_ab:,} | completed: {n-n_ab:,} | "
          f"abandonment rate: {n_ab/n:.2%}")

    # ---- time-aware split ----
    train, test = time_split(sessions, a.test_size)
    num, cat, dropped = select_features(train)
    cols = num + cat
    Xtr, ytr, gtr = train[cols], train[TARGET].to_numpy(), train["user_id"].to_numpy()
    Xte, yte = test[cols], test[TARGET].to_numpy()
    print(f"Train: {len(train):,} sessions ({train.session_start.min()} -> {train.session_start.max()})")
    print(f"Test : {len(test):,} sessions ({test.session_start.min()} -> {test.session_start.max()})")
    print(f"Train abandonment {ytr.mean():.2%} | Test abandonment {yte.mean():.2%}")
    print("Features used:", cols, "| dropped (corr>0.95):", dropped)

    # ---- class imbalance: REAL distribution kept; models re-weighted, never resampled ----
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))   # XGBoost: negatives / positives
    if demo_mode:
        # Small search spaces make a local smoke-test quicker.
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                                    {"clf__C": [0.1, 1.0]}),
            "Random Forest": (RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                                     random_state=a.seed),
                              {"clf__n_estimators": [50], "clf__max_depth": [6, None],
                               "clf__min_samples_leaf": [1, 5]}),
            "XGBoost": (XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=-1,
                                      random_state=a.seed, scale_pos_weight=spw),
                        {"clf__n_estimators": [50], "clf__max_depth": [3, 5],
                         "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8],
                         "clf__colsample_bytree": [0.8]}),
        }
    else:
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced"),
                                    {"clf__C": [0.01, 0.1, 1, 10]}),
            "Random Forest": (RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                                     random_state=a.seed),
                              {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, 16, None],
                               "clf__min_samples_leaf": [1, 5, 10, 20]}),
            "XGBoost": (XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=-1,
                                      random_state=a.seed, scale_pos_weight=spw),
                        {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, 7],
                         "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.9, 1.0],
                         "clf__colsample_bytree": [0.7, 0.9, 1.0]}),
        }
    # tuning subsample (keeps RF/XGB search affordable); final fit uses ALL training sessions
    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(Xtr))
    if len(idx) > a.tune_rows:
        idx = np.sort(rng.choice(idx, a.tune_rows, replace=False))
    cv = StratifiedGroupKFold(a.cv_folds, shuffle=True, random_state=a.seed)   # same user never in 2 folds

    rows, fitted, probas = [], {}, {}
    for name, (clf, grid) in models.items():
        pipe = Pipeline([("prep", preprocessor(num, cat)), ("clf", clf)])
        requested_iter = min(a.n_iter, 2) if demo_mode else a.n_iter
        n_iter = int(min(requested_iter, np.prod([len(v) for v in grid.values()])))
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="roc_auc", cv=cv,
                                    random_state=a.seed, n_jobs=-1)
        search.fit(Xtr.iloc[idx], ytr[idx], groups=gtr[idx])
        best = clone(pipe).set_params(**search.best_params_).fit(Xtr, ytr)
        proba = best.predict_proba(Xte)[:, 1]
        m, pred = score(yte, proba)
        rows.append({"model": name, **m, "cv_roc_auc": search.best_score_,
                     "best_params": json.dumps({k.replace("clf__", ""): v for k, v in
                                                search.best_params_.items()}, default=str)})
        fitted[name], probas[name] = best, (proba, pred)
        print(f"{name:20s} CV-AUC={search.best_score_:.4f} | TEST " +
              " ".join(f"{k}={m[k]:.4f}" for k in METRICS))

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/model_comparison.csv", index=False)
    res[["model"] + METRICS].to_csv(f"{OUT}/powerbi/model_metrics.csv", index=False)
    best_name = res.sort_values("cv_roc_auc", ascending=False).iloc[0]["model"]   # chosen on CV, not test
    print("Best model (by CV ROC-AUC):", best_name)

    # ---- plots ----
    names = list(probas)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, nme in enumerate(names):
        sns.heatmap(confusion_matrix(yte, probas[nme][1]), annot=True, fmt="d", cmap="Blues", ax=ax[i],
                    xticklabels=["Completed", "Abandoned"], yticklabels=["Completed", "Abandoned"])
        ax[i].set(title=nme, xlabel="Predicted", ylabel="Actual")
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for nme in names:
        fpr, tpr, _ = roc_curve(yte, probas[nme][0])
        plt.plot(fpr, tpr, label=f"{nme} (AUC={roc_auc_score(yte, probas[nme][0]):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/roc_curves.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for nme in names:
        p, r, _ = precision_recall_curve(yte, probas[nme][0])
        plt.plot(r, p, label=f"{nme} (AP={average_precision_score(yte, probas[nme][0]):.3f})")
    plt.axhline(yte.mean(), color="k", ls="--", label=f"No-skill ({yte.mean():.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves (test set)")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/pr_curves.png", dpi=150); plt.close()

    long = res.melt(id_vars="model", value_vars=METRICS, var_name="metric", value_name="score")
    plt.figure(figsize=(10, 4.5))
    sns.barplot(data=long, x="metric", y="score", hue="model")
    plt.ylim(0, 1); plt.title("Model comparison (test set)"); plt.tight_layout()
    plt.savefig(f"{OUT}/model_comparison.png", dpi=150); plt.close()

    # ---- explainability: permutation importance on original columns ----
    sub = Xte.sample(min(len(Xte), a.perm_rows), random_state=a.seed)
    imp = permutation_importance(fitted[best_name], sub, yte[Xte.index.get_indexer(sub.index)],
                                 scoring="roc_auc", n_repeats=5, random_state=a.seed, n_jobs=-1)
    fi = pd.DataFrame({"feature": cols, "importance": imp.importances_mean, "std": imp.importances_std}
                      ).sort_values("importance", ascending=False)
    fi.to_csv(f"{OUT}/feature_importance.csv", index=False)
    fi.to_csv(f"{OUT}/powerbi/feature_importance.csv", index=False)
    plt.figure(figsize=(7, 6))
    sns.barplot(data=fi.head(15), y="feature", x="importance", color="#3b82f6")
    plt.title(f"Permutation importance - {best_name}\n(drop in test ROC-AUC)"); plt.tight_layout()
    plt.savefig(f"{OUT}/feature_importance.png", dpi=150); plt.close()

    # ---- artefacts ----
    run = {**meta, "train_sessions": len(train), "test_sessions": len(test),
           "train_start": str(train.session_start.min()), "train_end": str(train.session_start.max()),
           "test_start": str(test.session_start.min()), "test_end": str(test.session_start.max()),
           "train_abandonment_rate": float(ytr.mean()), "test_abandonment_rate": float(yte.mean()),
           "best_model": best_name, "features": cols, "numeric": num, "categorical": cat,
           "dropped_correlated": dropped, "scale_pos_weight_xgb": spw, "seed": a.seed,
           "split": "time-ordered: earliest %.0f%% of sessions train, latest %.0f%% test" %
                    (100 * (1 - a.test_size), 100 * a.test_size),
           "cv": f"StratifiedGroupKFold({a.cv_folds}) grouped by user_id, tuned on {len(idx):,} train rows",
           "runtime_min": round((time.time() - t0) / 60, 1)}
    json.dump(run, open(f"{OUT}/run_metadata.json", "w"), indent=2)
    joblib.dump({"model": fitted[best_name], "name": best_name, "features": cols, "numeric": num,
                 "categorical": cat, "meta": run,
                 "main_categories": sorted(sessions["main_category"].dropna().unique().tolist())},
                f"{OUT}/best_model.joblib")
    out_te = test[["session_start"] + cols].copy()
    out_te["actual_abandoned"] = yte
    out_te["predicted_prob_abandon"] = probas[best_name][0]
    out_te["predicted_abandoned"] = probas[best_name][1]
    out_te.sample(min(len(out_te), 100_000), random_state=a.seed).to_csv(
        f"{OUT}/powerbi/session_predictions.csv", index=False)

    report_source = "SYNTHETIC DEMO DATA - NOT FOR RESEARCH REPORTING" if demo_mode else "REAL REES46 data"
    md = f"""# SmartCart results (auto-generated from this run - {report_source})

* Files: {', '.join(meta['files'])}
* User sampling fraction (of REAL users): {meta['sample_frac']}
* Events scanned: {meta['events_scanned']:,} | sessions: {meta['sessions_total']:,} | cart sessions: {meta['cart_sessions']:,}
* Prediction point: first add-to-cart + {meta['window_minutes']} min; sessions already purchased by then are removed
* Modelling sessions: {n:,} (abandoned {n_ab:,} / completed {n-n_ab:,} = {n_ab/n:.2%} abandonment)
* Split: {run['split']}
  * Train: {len(train):,} sessions, {run['train_start']} -> {run['train_end']}, abandonment {ytr.mean():.2%}
  * Test : {len(test):,} sessions, {run['test_start']} -> {run['test_end']}, abandonment {yte.mean():.2%}
* CV: {run['cv']}
* Imbalance: real distribution kept; class_weight='balanced' (LR), 'balanced_subsample' (RF), scale_pos_weight={spw:.3f} (XGB)
* Metrics: positive class = abandoned; decision threshold 0.5; best model chosen on CV ROC-AUC: **{best_name}**

## Test-set performance
{md_table(res[['model'] + METRICS + ['cv_roc_auc']])}

## Top factors (permutation importance, {best_name})
{md_table(fi.head(10))}
"""
    open(f"{OUT}/results_summary.md", "w").write(md)
    print(f"\nWrote results to {OUT}/ (see results_summary.md). Total {run['runtime_min']} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train SmartCart models on real REES46 data or synthetic demo sessions")
    ap.add_argument("--data-dir", default="data/raw", help="folder with the Kaggle REES46 CSV files")
    ap.add_argument("--files", nargs="*", help="specific CSVs, e.g. 2019-Oct.csv 2019-Nov.csv")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="fraction of REAL users sampled from the Kaggle data (NOT synthetic)")
    ap.add_argument("--window-minutes", type=float, default=5.0)
    ap.add_argument("--features", help="reuse a table made by src.preprocess instead of re-reading raw CSVs")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=6, help="random-search candidates per model")
    ap.add_argument("--tune-rows", type=int, default=200_000, help="train rows used for tuning")
    ap.add_argument("--perm-rows", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true",
                    help="run a synthetic pipeline smoke-test without downloading REES46")
    ap.add_argument("--demo-rows", type=int, default=1000,
                    help="number of synthetic sessions for --demo (minimum 200)")
    main(ap.parse_args())
