# SmartCart: E-Commerce Cart Abandonment Prediction

Predicts whether a shopping session that has reached the cart will be **abandoned**, using real
clickstream data from the **REES46 eCommerce Behavior Dataset**. The code never generates data for training
or evaluation.

## 1. Problem Statement
Many online shoppers add products to their cart and leave without paying. Each abandoned cart is lost
revenue, and a retailer that can spot a likely abandonment while the shopper is still on the site can intervene
(reminder, offer, recommendation) instead of finding out afterwards.

## 2. Objective
Given what a shopper has done in a session up to a defined **prediction point**, estimate the probability that
the session ends **without a purchase**, and turn that probability into a risk level (LOW / MEDIUM / HIGH) and a
suggested business action. Three models are compared: Logistic Regression (baseline), Random Forest, XGBoost.

## 3. Dataset
**REES46 – eCommerce behavior data from multi-category store** (Kechinov, REES46 Marketing Platform).
Kaggle: <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>
(`mkechinov/ecommerce-behavior-data-from-multi-category-store`).

Real behaviour of an online store, about 285 million events from Oct 2019 to Apr 2020, one CSV per month. It
is **very large** (many GB per month), so the pipeline reads it in chunks and supports sampling (Section 4).

| Column | Meaning |
|---|---|
| `event_time` | UTC time of the event |
| `event_type` | `view`, `cart`, `remove_from_cart`, `purchase` |
| `product_id`, `category_id` | product / category identifiers |
| `category_code` | category taxonomy (often missing) |
| `brand` | brand (often missing; not used) |
| `price` | product price |
| `user_id` | permanent user id |
| `user_session` | session id (changes after a long pause) |

The dataset has **no device column**, so none is used.

## 4. Dataset Setup
1. Download the dataset from Kaggle (website, or the Kaggle CLI):
   ```bash
   pip install kaggle          # needs your API token in ~/.kaggle/kaggle.json
   kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store -p data/raw --unzip
   ```
2. Extract the CSV files if you downloaded a zip.
3. Place them in `data/raw/`:
   ```text
   data/raw/2019-Oct.csv  2019-Nov.csv  2019-Dec.csv  2020-Jan.csv  2020-Feb.csv  2020-Mar.csv  2020-Apr.csv
   ```
   You can use only some of them.
4. **Do not commit the CSVs.** `.gitignore` already excludes `data/raw/*.csv`, `data/processed/*` and `outputs/*`.

### Run
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# train on REAL data (all CSVs found in data/raw, 2% of users sampled)
python -m src.train --data-dir data/raw --sample-frac 0.02

# or choose files explicitly
python -m src.train --data-dir data/raw --files 2019-Oct.csv 2019-Nov.csv --sample-frac 0.05

# optional: build the session table once, then reuse it
python -m src.preprocess --data-dir data/raw --sample-frac 0.02
python -m src.train --features data/processed/session_features.csv

streamlit run app.py
python -m pytest tests -q          # code-correctness tests (hand-written mini fixtures)
```
**`--sample-frac` samples real users from the Kaggle files** (all events of a sampled user are kept, so sessions
and purchase history stay complete). It does not generate or simulate any data. `--sample-frac 1.0` (default)
uses everything, which needs a lot of RAM/time on the full dataset. Start at `0.01`–`0.05`.

## 5. Project Architecture
```text
Raw REES46 Events (data/raw/*.csv)
        ↓  chunked reads, only needed columns, user-level sampling
Data Cleaning            (valid event types, parse times, price ≥ 0)
        ↓
Session Formation        (pass 1: one row per session; pass 2: events of cart sessions)
        ↓
Leakage-Safe Feature Engineering   (only events up to the prediction point)
        ↓
Session-Level Dataset    (data/processed/session_features.csv + .meta.json)
        ↓
Time-ordered Train/Test Split
        ↓
Logistic Regression | Random Forest | XGBoost   (grouped CV + random search)
        ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix)
        ↓
Best Model (chosen on CV ROC-AUC)  →  outputs/best_model.joblib
        ↓
Abandonment Probability  →  Risk Level  →  Business Intervention   (Streamlit app)
```
```text
src/data_loader.py   file discovery, chunked reading, user sampling, session summary, cart-session events
src/features.py      target, prediction point, leakage-safe features, user-history features
src/preprocess.py    CLI that builds the session table
src/train.py         split, feature selection, tuning, evaluation, plots, artefacts
app.py               Streamlit UI
tests/               unit tests for labels, leakage guard, user history and the prediction-point boundary
```

## 6. Target Variable
Computed per `user_session`; only sessions that reached the cart are kept.

| Session content | `abandoned` |
|---|---|
| has `cart`, **no** `purchase` | **1** (cart abandonment) |
| has `cart` and a `purchase` after the prediction point | **0** (purchase completed) |
| no `cart` event | not in the dataset |
| `purchase` at or before the prediction point | removed (outcome already known) |

## 7. Feature Engineering
All features use only events with `time ≤ prediction point` and never use `purchase` events.

| Feature | Description |
|---|---|
| `n_events`, `n_views`, `n_cart_adds`, `n_cart_removes` | event counts |
| `net_cart_additions` | adds − removals |
| `n_unique_products`, `n_unique_categories` | breadth of browsing |
| `avg_price`, `max_price` | price level of viewed/carted items |
| `cart_value` | sum of prices of cart additions |
| `removal_ratio`, `views_to_cart_ratio`, `cart_to_view_ratio` | behaviour ratios |
| `session_duration_sec`, `time_to_first_cart_sec` | time features |
| `avg_time_between_actions_sec`, `events_per_minute` | activity intensity |
| `start_hour`, `day_of_week`, `is_weekend` | temporal context (UTC) |
| `n_prev_sessions`, `n_prev_purchases`, `is_returning_user` | user history from **earlier** sessions only |
| `main_category` | most frequent top-level category (`unknown` if missing) |

Before training, constant columns and one of every highly correlated pair (|r| > 0.95, computed on the
training set) are dropped. The kept list is stored in `outputs/run_metadata.json`.
History features only see the files/sample you loaded: with a single month, users active earlier look "new".

## 8. Machine Learning Models
- **Logistic Regression** (baseline): linear model on scaled features, easy to interpret.
- **Random Forest**: many decision trees on bootstrap samples; captures non-linear interactions.
- **XGBoost**: gradient-boosted trees; usually strongest on tabular data.

Hyperparameters come from `RandomizedSearchCV` (ROC-AUC) on a training subsample (`--tune-rows`, default
200,000); the winner is then refit on **all** training sessions.

## 9. Evaluation Metrics
Positive class = **abandoned**; threshold 0.5.
- **Accuracy**: share of correct predictions (misleading if one class dominates).
- **Precision**: of sessions flagged as abandoning, how many really did.
- **Recall**: of the sessions that abandoned, how many were caught.
- **F1**: harmonic mean of precision and recall.
- **ROC-AUC**: ability to rank abandoners above purchasers (threshold-free).
- **PR-AUC (average precision)**: precision/recall trade-off, informative under imbalance.
- **Balanced accuracy** and the **confusion matrix** are also reported.

## 10. Class Imbalance
Most cart sessions end without a purchase, so abandonment is typically the **majority** class. The project keeps
the **real** class distribution (no over/under-sampling, no synthetic minority samples) and reports it. Models
are re-weighted: `class_weight="balanced"` (Logistic Regression), `"balanced_subsample"` (Random Forest) and
`scale_pos_weight = negatives/positives` (XGBoost). Because accuracy can look good by predicting "abandoned" for
everyone, judge models by recall, precision, F1, PR-AUC and ROC-AUC, and compare PR-AUC with the
no-skill line (the abandonment rate).

## 11. Data Leakage Prevention
- **Prediction point** = time of the session's **first `cart` event + 5 minutes** (`--window-minutes`).
- Features use only events at or before the prediction point; later views, carts, removals and the whole rest of
  the session are ignored.
- `purchase` events define the label only and are never inputs (a unit test checks that changing purchase
  prices leaves every feature unchanged).
- Sessions that purchased at or before the prediction point are removed.
- User-history features count only earlier sessions (strictly earlier start / earlier first purchase).
- Time-ordered split and user-grouped CV (Section 12/13 notes) keep test information out of training.

## 12. Results
**No results are stored in this repository.** Run the training command on your downloaded REES46 files; it
writes the real numbers to `outputs/results_summary.md` (files used, sampling fraction, session counts, class
distribution, split dates, metrics table, top features) plus:

`model_comparison.csv/.png`, `confusion_matrices.png`, `roc_curves.png`, `pr_curves.png`,
`feature_importance.csv/.png`, `run_metadata.json`, `best_model.joblib`, `powerbi/*.csv`.

Paste the contents of `outputs/results_summary.md` here once you have run it and state exactly which files
and `--sample-frac` you used.

Method: **time-ordered split** (earliest 80% of sessions train, latest 20% test); tuning uses
`StratifiedGroupKFold` grouped by `user_id` so one user is never in both a training and a validation fold; the best
model is selected on CV ROC-AUC, never on the test set.

## 13. Explainability
**Permutation importance** shuffles one feature at a time on held-out test sessions and measures how much ROC-AUC
drops. Big drop = the model relies on that feature. It describes the model's behaviour and association, not
causes, and correlated features can share importance.

## 14. Streamlit Application
`streamlit run app.py`. Enter session behaviour up to the prediction point and get:
```text
Abandonment Probability: xx.x%
Risk Level: HIGH / MEDIUM / LOW      (thresholds adjustable in the sidebar)
```
There is also batch scoring of a CSV and a results tab. The app only loads a model whose metadata says it was
trained on REES46 data. If `outputs/best_model.joblib` is missing it tells you to train first.

## 15. Business Use Case
Score sessions at the prediction point and act by risk: **HIGH** → cart-reminder email/push, free-shipping or
limited-time discount; **MEDIUM** → recommendations, reviews, low-stock messages; **LOW** → no discount (protects
margin). Returning users with purchase history can get personalised messages. Measure with an A/B test.

## 16. Research References
1. Rausch, T. M., Derra, N. D., & Wolf, L. (2022). Predicting online shopping cart abandonment with machine
   learning approaches. *International Journal of Market Research, 64*(1). https://doi.org/10.1177/1470785320972526
2. Kukar-Kinney, M., Scheinbaum, A. C., Orimoloye, L. O., et al. (2022). A model of online shopping cart
   abandonment: evidence from e-tail clickstream data. *Journal of the Academy of Marketing Science, 50*, 961–980.
   https://doi.org/10.1007/s11747-022-00857-8
3. Prediction of cart abandonment using imbalanced clickstream data in online shopping. *Electronic Commerce
   Research*, Springer. https://doi.org/10.1007/s10660-025-10021-3
4. Categorizing Online Shopping Behavior from Cosmetics to Electronics: An Analytical Framework.
   arXiv:2010.02503. https://arxiv.org/abs/2010.02503
5. Kechinov, M. REES46 eCommerce behavior data from multi category store (Kaggle).
   https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
   https://doi.org/10.1145/2939672.2939785
7. Breiman, L. (2001). Random Forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324

Please check the details of each reference yourself before citing it in your report.

## 17. Limitations
- **Size/compute**: the full dataset needs lots of RAM and time; most experiments use a user sample, and results
  depend on it. Tuning uses a training subsample.
- **Available features**: no device, demographics, referrer, shipping cost or stock data; `brand`/`category_code` are
  often missing.
- **Label noise**: a user may buy in a *later* session or on another channel; that session is still labelled abandoned.
  Sessions near the end of a file can be truncated.
- **User history** is limited to the loaded files/sample.
- **Prediction point** is a design choice (5 min after first cart); results change with it. Quick buyers (purchase
  inside the window) are excluded, so the model targets sessions that continue past that point.
- **Association, not causation**: importance scores do not show that an intervention will change behaviour.
- Intent is only partly visible in clicks, so performance has a ceiling.

## 18. Future Work
Sequential models (LSTM / Transformer) on raw event sequences; real-time scoring at several prediction points;
online learning and drift monitoring; cross-domain validation on other stores; uplift modelling and A/B tests for
personalised interventions; cost-sensitive threshold tuning.
