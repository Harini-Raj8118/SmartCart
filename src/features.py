"""Leakage-safe session features for REAL REES46 events.

TARGET   abandoned = 1  : session has a `cart` event and NO `purchase` event
         abandoned = 0  : session has a `cart` event and a `purchase` AFTER the prediction point
         Sessions without a cart event are not part of the dataset.

PREDICTION POINT  t_pred = time of the session's FIRST `cart` event + WINDOW_MINUTES (default 5).
  * Every feature uses only events with time <= t_pred (never anything later).
  * `purchase` events are never features; they only define the label.
  * Sessions that already purchased at or before t_pred are dropped (outcome already known).
  * User history features use only EARLIER sessions (they started / purchased before this session started).
"""
import numpy as np
import pandas as pd

NUMERIC = [
    "n_events", "n_views", "n_cart_adds", "n_cart_removes", "net_cart_additions",
    "n_unique_products", "n_unique_categories", "avg_price", "max_price", "cart_value",
    "removal_ratio", "views_to_cart_ratio", "cart_to_view_ratio",
    "session_duration_sec", "time_to_first_cart_sec", "avg_time_between_actions_sec",
    "events_per_minute", "start_hour", "day_of_week", "is_weekend",
    "n_prev_sessions", "n_prev_purchases", "is_returning_user",
]
CATEGORICAL = ["main_category"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "abandoned"
ID_COLS = ["session_key", "user_id", "session_start"]
DEFAULT_WINDOW_MIN = 5.0


def count_before(ev_user, ev_time, q_user, q_time):
    """For every query (user, time): number of events of the SAME user strictly earlier."""
    e = pd.DataFrame({"u": ev_user, "t": ev_time, "flag": 1, "i": -1}).dropna(subset=["t"])
    q = pd.DataFrame({"u": q_user, "t": q_time, "flag": 0, "i": np.arange(len(q_user))})
    both = pd.concat([e, q], ignore_index=True).sort_values(["u", "t", "flag"], kind="stable")
    both["c"] = both.groupby("u")["flag"].cumsum()          # queries sort before events at equal time
    res = np.zeros(len(q_user), dtype="int64")
    qq = both[both["flag"] == 0]
    res[qq["i"].to_numpy()] = qq["c"].to_numpy()
    return res


def build_features(events: pd.DataFrame, summary: pd.DataFrame,
                   window_minutes: float = DEFAULT_WINDOW_MIN) -> pd.DataFrame:
    """events: cart-session events (columns from data_loader). summary: pass-1 session table."""
    ev = events.sort_values(["session_key", "t"], kind="stable")
    win = pd.Timedelta(minutes=window_minutes)

    first_cart = ev[ev["event_type"] == "cart"].groupby("session_key")["t"].min().rename("t_cart")
    first_buy = ev[ev["event_type"] == "purchase"].groupby("session_key")["t"].min().rename("t_buy")
    s = pd.concat([first_cart, first_buy], axis=1).dropna(subset=["t_cart"])
    s["t_pred"] = s["t_cart"] + win

    # drop sessions whose outcome was already known at the prediction point
    s = s[s["t_buy"].isna() | (s["t_buy"] > s["t_pred"])]
    s[TARGET] = s["t_buy"].isna().astype(int)

    # ---- LEAKAGE GUARD: feature events = not purchase AND time <= t_pred ----
    f = ev.join(s[["t_pred"]], on="session_key", how="inner")
    f = f[(f["event_type"] != "purchase") & (f["t"] <= f["t_pred"])].copy()
    f["is_view"] = (f["event_type"] == "view")
    f["is_cart"] = (f["event_type"] == "cart")
    f["is_rem"] = (f["event_type"] == "remove_from_cart")
    f["cart_price"] = f["price"].where(f["is_cart"], 0.0)
    f["gap"] = f.groupby("session_key")["t"].diff().dt.total_seconds()

    g = f.groupby("session_key")
    X = pd.DataFrame({
        "n_events": g.size(),
        "n_views": g["is_view"].sum(), "n_cart_adds": g["is_cart"].sum(), "n_cart_removes": g["is_rem"].sum(),
        "n_unique_products": g["product_id"].nunique(),
        "n_unique_categories": g["category_id"].nunique(),
        "avg_price": g["price"].mean(), "max_price": g["price"].max(),
        "cart_value": g["cart_price"].sum(),
        "avg_time_between_actions_sec": g["gap"].mean(),
        "session_start": g["t"].min(), "last_t": g["t"].max(),
        "user_id": g["user_id"].first(),
        "main_category": g["main_category"].agg(lambda x: x.mode().iat[0]),
    })
    X = X.join(s[[TARGET, "t_cart"]], how="inner")
    X["net_cart_additions"] = X["n_cart_adds"] - X["n_cart_removes"]
    X["removal_ratio"] = X["n_cart_removes"] / X["n_cart_adds"].replace(0, np.nan)
    X["views_to_cart_ratio"] = X["n_views"] / X["n_cart_adds"].replace(0, np.nan)
    X["cart_to_view_ratio"] = X["n_cart_adds"] / (X["n_views"] + 1)
    X["session_duration_sec"] = (X["last_t"] - X["session_start"]).dt.total_seconds()
    X["time_to_first_cart_sec"] = (X["t_cart"] - X["session_start"]).dt.total_seconds()
    X["events_per_minute"] = X["n_events"] / (X["session_duration_sec"] / 60 + 1)
    X["start_hour"] = X["session_start"].dt.hour
    X["day_of_week"] = X["session_start"].dt.dayofweek
    X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)
    for c in ["removal_ratio", "views_to_cart_ratio", "avg_time_between_actions_sec"]:
        X[c] = X[c].fillna(0)

    # ---- user history from EARLIER sessions only (no leakage) ----
    X = X.reset_index()
    X["n_prev_sessions"] = count_before(summary["user_id"].to_numpy(), summary["start"].to_numpy(),
                                        X["user_id"].to_numpy(), X["session_start"].to_numpy())
    bought = summary.dropna(subset=["first_purchase"])
    X["n_prev_purchases"] = count_before(bought["user_id"].to_numpy(), bought["first_purchase"].to_numpy(),
                                         X["user_id"].to_numpy(), X["session_start"].to_numpy())
    X["is_returning_user"] = (X["n_prev_sessions"] > 0).astype(int)

    X[["n_views", "n_cart_adds", "n_cart_removes", "net_cart_additions"]] = \
        X[["n_views", "n_cart_adds", "n_cart_removes", "net_cart_additions"]].astype(int)
    return X[ID_COLS + FEATURES + [TARGET]].reset_index(drop=True)
