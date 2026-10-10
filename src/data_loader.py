"""Memory-safe access to the REAL REES46 monthly CSVs in data/raw/.

Two chunked passes per run (the raw files are tens of millions of rows each):
  pass 1  -> one slim row per session (user, start, first purchase, has_cart) for ALL selected files
  pass 2  -> the full event rows, but ONLY for sessions that reached the cart
`sample_frac` samples USERS from the real data (all of a sampled user's events are kept, so sessions
and purchase history stay complete). It never generates data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ["event_time", "event_type", "product_id", "category_id", "category_code",
            "brand", "price", "user_id", "user_session"]
USECOLS = ["event_time", "event_type", "product_id", "category_id", "category_code",
           "price", "user_id", "user_session"]       # brand is not used by any feature
VALID_EVENTS = {"view", "cart", "remove_from_cart", "purchase"}
TIME_FMT = "%Y-%m-%d %H:%M:%S UTC"


def discover_files(data_dir="data/raw", files=None):
    """Find REES46 CSVs. `files` = optional explicit names (e.g. ['2019-Oct.csv'])."""
    d = Path(data_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} does not exist. See README section 'Dataset Setup'.")
    paths = [d / f for f in files] if files else sorted(d.glob("*.csv"))
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Not found: {missing}")
    if not paths:
        raise FileNotFoundError(f"No CSV files in {d}. Download REES46 from Kaggle first (see README).")
    for p in paths:                                   # fail loudly on anything that is not REES46
        cols = pd.read_csv(p, nrows=0).columns
        gone = [c for c in REQUIRED if c not in cols]
        if gone:
            raise ValueError(f"{p.name} is not a REES46 file (missing columns {gone}).")
    return paths


def parse_time(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, format=TIME_FMT, utc=True)
    except (ValueError, TypeError):
        return pd.to_datetime(s, utc=True, errors="coerce")


def session_key(s: pd.Series) -> np.ndarray:
    """64-bit hash of the session id string (stable across passes, far smaller than strings)."""
    return pd.util.hash_pandas_object(s.astype(str), index=False).to_numpy()


def keep_user(user_id: pd.Series, frac: float) -> np.ndarray:
    """Deterministic user-level sample of the REAL data."""
    if frac >= 1.0:
        return np.ones(len(user_id), dtype=bool)
    h = (user_id.to_numpy().astype("uint64") * np.uint64(2654435761)) % np.uint64(10_000)
    return h < np.uint64(int(frac * 10_000))


def _chunks(path, chunksize):
    for ch in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize,
                          dtype={"event_type": "category", "category_code": "string"}):
        ch = ch[ch["event_type"].isin(VALID_EVENTS)]
        ch = ch.dropna(subset=["user_id", "user_session", "event_time"])
        yield ch


def build_session_summary(paths, sample_frac=1.0, chunksize=2_000_000, log=print):
    """PASS 1: one row per session over all selected files."""
    parts, n_rows = [], 0
    for p in paths:
        log(f"[pass 1] scanning {p.name}")
        for ch in _chunks(p, chunksize):
            ch = ch[keep_user(ch["user_id"], sample_frac)]
            if ch.empty:
                continue
            n_rows += len(ch)
            t = parse_time(ch["event_time"])
            tmp = pd.DataFrame({
                "session_key": session_key(ch["user_session"]),
                "user_id": ch["user_id"].to_numpy(),
                "t": t.to_numpy(),
                "t_cart": t.where(ch["event_type"].to_numpy() == "cart").to_numpy(),
                "t_buy": t.where(ch["event_type"].to_numpy() == "purchase").to_numpy(),
            })
            parts.append(tmp.groupby("session_key").agg(
                user_id=("user_id", "first"), start=("t", "min"), end=("t", "max"),
                first_cart=("t_cart", "min"), first_purchase=("t_buy", "min")).reset_index())
            if len(parts) >= 20:                      # keep memory flat
                parts = [_reduce(parts)]
    if not parts:
        raise ValueError("No events left after sampling; increase --sample-frac.")
    out = _reduce(parts)
    out["has_cart"] = out["first_cart"].notna()
    log(f"[pass 1] {n_rows:,} events -> {len(out):,} sessions "
        f"({int(out['has_cart'].sum()):,} reached the cart)")
    return out, n_rows


def _reduce(parts):
    return pd.concat(parts).groupby("session_key").agg(
        user_id=("user_id", "first"), start=("start", "min"), end=("end", "max"),
        first_cart=("first_cart", "min"), first_purchase=("first_purchase", "min")).reset_index()


def iter_cart_session_events(paths, cart_keys, sample_frac=1.0, chunksize=2_000_000, log=print):
    """PASS 2: yield, per file, the events of cart-reaching sessions only."""
    cart_keys = np.sort(np.asarray(cart_keys, dtype="uint64"))
    for p in paths:
        log(f"[pass 2] extracting cart-session events from {p.name}")
        parts = []
        for ch in _chunks(p, chunksize):
            ch = ch[keep_user(ch["user_id"], sample_frac)]
            if ch.empty:
                continue
            k = session_key(ch["user_session"])
            m = np.isin(k, cart_keys, assume_unique=False)
            if not m.any():
                continue
            ch = ch[m]
            main_cat = ch["category_code"].str.split(".").str[0].fillna("unknown")
            parts.append(pd.DataFrame({
                "session_key": k[m], "user_id": ch["user_id"].to_numpy(),
                "t": parse_time(ch["event_time"]).to_numpy(),
                "event_type": ch["event_type"].astype(str).to_numpy(),
                "product_id": ch["product_id"].to_numpy(),
                "category_id": ch["category_id"].to_numpy(),
                "main_category": main_cat.to_numpy(),
                "price": pd.to_numeric(ch["price"], errors="coerce").fillna(0).clip(lower=0).to_numpy(),
            }))
        if parts:
            yield p.name, pd.concat(parts, ignore_index=True)
