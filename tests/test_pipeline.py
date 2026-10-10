"""Code-correctness tests on tiny HAND-WRITTEN event tables. They validate logic only; no result is reported from them."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import preprocess
from src.features import FEATURES, TARGET

T0 = pd.Timestamp("2019-10-01 10:00:00")


def ev(sess, user, minutes, etype, price=100.0, product=1, cat="electronics.smartphone"):
    t = (T0 + pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S UTC")
    return dict(event_time=t, event_type=etype, product_id=product, category_id=10 + product,
                category_code=cat, brand="x", price=price, user_id=user, user_session=sess)


def write(tmp_path, rows):
    d = tmp_path / "raw"; d.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "2019-Oct.csv", index=False)
    return d


def run(tmp_path, rows, window=5.0):
    d = write(tmp_path, rows)
    X, meta = preprocess(str(d), None, 1.0, window, 1000, str(tmp_path / "f.csv"), log=lambda *_: None)
    return X.set_index("session_key"), meta


def rows_basic(purchase_price=100.0):
    r = []
    # A: purchase AFTER the 5-min window -> completed (0)
    r += [ev("A", 1, 0, "view"), ev("A", 1, 1, "cart", 50, 2), ev("A", 1, 3, "view", 70, 3),
          ev("A", 1, 10, "purchase", purchase_price, 2)]
    # B: never purchases -> abandoned (1)
    r += [ev("B", 2, 0, "view"), ev("B", 2, 1, "cart", 30, 2), ev("B", 2, 2, "view", 20, 3)]
    # C: purchased inside the window -> already known, must be DROPPED
    r += [ev("C", 3, 0, "view"), ev("C", 3, 1, "cart", 30, 2), ev("C", 3, 3, "purchase", 30, 2)]
    # D: only views -> not a cart session, EXCLUDED
    r += [ev("D", 4, 0, "view"), ev("D", 4, 1, "view", 20, 3)]
    # E: events after the prediction point must be ignored
    r += [ev("E", 5, 0, "view"), ev("E", 5, 1, "cart", 40, 2), ev("E", 5, 6.5, "remove_from_cart", 40, 2),
          ev("E", 5, 7, "view", 999, 4), ev("E", 5, 30, "cart", 500, 5)]
    return r


def test_labels_and_session_filtering(tmp_path):
    X, meta = run(tmp_path, rows_basic())
    assert len(X) == 3                                   # A, B, E only (C dropped, D excluded)
    assert sorted(X[TARGET].tolist()) == [0, 1, 1]       # A completed, B + E abandoned
    assert meta["synthetic"] is False and meta["abandoned"] == 2 and meta["completed"] == 1


def test_no_future_information_in_features(tmp_path):
    X, _ = run(tmp_path, rows_basic())
    e = X[(X["n_cart_removes"] == 0) & (X["n_events"] == 2) & (X["user_id"] == 5)]
    assert len(e) == 1                                   # E: view + cart only; the t=6/7/30 events are ignored
    assert e["max_price"].iloc[0] == 100 and e["cart_value"].iloc[0] == 40   # 999-price view / 500 cart are in the future
    assert e["session_duration_sec"].iloc[0] == 60


def test_purchase_never_changes_features(tmp_path):
    a, _ = run(tmp_path, rows_basic(purchase_price=100.0))
    b, _ = run(tmp_path, rows_basic(purchase_price=999999.0))
    pd.testing.assert_frame_equal(a[FEATURES].sort_index(), b[FEATURES].sort_index())


def test_user_history_uses_only_earlier_sessions(tmp_path):
    r = [ev("P1", 9, 0, "view"), ev("P1", 9, 1, "cart"), ev("P1", 9, 20, "purchase"),   # earlier, bought
         ev("P2", 9, 60, "view"), ev("P2", 9, 61, "cart"),                              # current, abandons
         ev("P3", 9, 180, "view"), ev("P3", 9, 181, "cart"), ev("P3", 9, 200, "purchase")]  # later
    X, _ = run(tmp_path, r)
    p2 = X.sort_values("session_start").iloc[1]
    assert p2["n_prev_sessions"] == 1 and p2["n_prev_purchases"] == 1 and p2["is_returning_user"] == 1
    first = X.sort_values("session_start").iloc[0]
    assert first["n_prev_sessions"] == 0 and first["n_prev_purchases"] == 0


def test_rejects_non_rees46_file(tmp_path):
    d = tmp_path / "raw"; d.mkdir()
    pd.DataFrame({"a": [1]}).to_csv(d / "x.csv", index=False)
    with pytest.raises(ValueError):
        preprocess(str(d), None, 1.0, 5.0, 1000, str(tmp_path / "f.csv"), log=lambda *_: None)


def test_prediction_point_boundary(tmp_path):
    r = [ev("Z", 7, 0, "view"), ev("Z", 7, 1, "cart"), ev("Z", 7, 6, "view"), ev("Z", 7, 6.5, "view")]
    X, _ = run(tmp_path, r)
    assert X["n_events"].iloc[0] == 3        # view, cart, view@t_pred ; the later view is excluded
