"""Create a reproducible, user-level sample from a real REES46 CSV.

Run from the project root:
    py scripts/create_200mb_sample.py --input data/raw/2019-Oct.csv

This samples whole users (all their rows in the selected input file), writes
a smaller raw-events CSV, then asks the existing SmartCart preprocessing
pipeline to validate whether both target classes are represented.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_COLUMNS = {
    "event_time", "event_type", "product_id", "category_id",
    "category_code", "brand", "price", "user_id", "user_session",
}


def stable_user_score(user_id: str, seed: int) -> int:
    """Return a deterministic 64-bit score for a user ID."""
    value = f"{seed}:{user_id}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def sample_csv(input_path: Path, output_path: Path, target_mb: float,
               seed: int, chunksize: int) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if target_mb <= 0:
        raise ValueError("--target-mb must be positive.")

    source_bytes = input_path.stat().st_size
    target_bytes = int(target_mb * 1_000_000)
    # Estimate an initial user inclusion rate from the target byte ratio.
    # User row counts vary, so this is an approximate size target, not a guarantee.
    fraction = min(1.0, target_bytes / max(source_bytes, 1))
    threshold = int(fraction * (2**64 - 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    selected_rows = 0
    selected_users = set()
    seen_users = set()
    columns = None

    try:
        reader = pd.read_csv(
            input_path, chunksize=chunksize, dtype={"user_id": "string"},
            low_memory=False
        )
        for chunk in reader:
            if columns is None:
                columns = list(chunk.columns)
                missing = REQUIRED_COLUMNS - set(columns)
                if missing:
                    raise ValueError(f"CSV is missing required REES46 columns: {sorted(missing)}")
            total_rows += len(chunk)
            ids = chunk["user_id"].astype("string")
            # The same user receives the same score in every chunk.
            scores = ids.map(
                lambda x: stable_user_score(str(x), seed)
                if pd.notna(x) else stable_user_score("__MISSING_USER__", seed)
            )
            keep = scores <= threshold
            chosen = chunk.loc[keep]
            if not chosen.empty:
                chosen.to_csv(
                    output_path, mode="a", index=False,
                    header=not output_path.exists()
                )
                selected_rows += len(chosen)
                selected_users.update(chosen["user_id"].dropna().astype(str).unique())
            if total_rows % max(chunksize, 1) == 0:
                print(f"Read {total_rows:,} rows; selected {selected_rows:,} rows...")

    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise

    if not output_path.exists() or selected_rows == 0:
        raise RuntimeError(
            "Sampling selected no rows. Try a larger target size or a different input file."
        )

    actual_bytes = output_path.stat().st_size
    result = {
        "dataset": "REES46 real raw events",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "seed": seed,
        "sampling_method": "deterministic SHA-256 user-level hash threshold",
        "estimated_user_inclusion_fraction": fraction,
        "source_size_mb_decimal": round(source_bytes / 1_000_000, 3),
        "target_size_mb_decimal": target_mb,
        "actual_size_mb_decimal": round(actual_bytes / 1_000_000, 3),
        "source_rows_read": total_rows,
        "sample_rows_retained": selected_rows,
        "selected_unique_users": len(selected_users),
        "note": (
            "Size is approximate because users have different event counts. "
            "All rows for selected user IDs in this input file are retained."
        ),
    }
    metadata_path = Path(str(output_path) + ".sample.json")
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSampling metadata written to: {metadata_path}")
    return result


def validate_with_project(sample_path: Path, chunksize: int, window_minutes: int) -> None:
    """Run the existing preprocessor; do not fabricate or rebalance labels."""
    from src.preprocess import preprocess
    out_path = ROOT / "data" / "processed" / "session_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("\nRunning SmartCart preprocessing on the sample...")
    X, meta = preprocess(
        data_dir=str(sample_path.parent),
        files=[sample_path.name],
        sample_frac=1.0,
        window_minutes=window_minutes,
        chunksize=chunksize,
        out=str(out_path),
    )
    target_name = "abandoned"
    if target_name not in X.columns:
        raise RuntimeError(
            f"Preprocessing did not produce '{target_name}'. "
            "Inspect src/features.py and src/preprocess.py."
        )
    counts = X[target_name].value_counts().to_dict()
    abandoned = int(counts.get(1, 0))
    completed = int(counts.get(0, 0))
    validation = {
        "sample_file": str(sample_path),
        "processed_features": str(out_path),
        "sessions_total": int(len(X)),
        "abandoned_sessions_class_1": abandoned,
        "completed_sessions_class_0": completed,
        "both_classes_present": abandoned > 0 and completed > 0,
        "preprocess_metadata": meta,
    }
    validation_path = ROOT / "data" / "processed" / "sample_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    print(json.dumps(validation, indent=2, default=str))
    print(f"\nValidation report: {validation_path}")
    if not validation["both_classes_present"]:
        raise RuntimeError(
            "Both target classes were not found in the processed sample. "
            "Try a larger sample or a different source month; do not artificially balance it."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/2019-Oct.csv",
                        help="Path to the original REES46 monthly CSV")
    parser.add_argument("--output", default="data/raw/rees46_sample_200mb.csv",
                        help="Path for the sampled raw-events CSV")
    parser.add_argument("--target-mb", type=float, default=200,
                        help="Approximate output size in decimal MB (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--skip-validation", action="store_true",
                        help="Only create the raw sample; do not preprocess it")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    sample_csv(input_path, output_path, args.target_mb, args.seed, args.chunksize)
    if not args.skip_validation:
        validate_with_project(output_path, args.chunksize, args.window_minutes)


if __name__ == "__main__":
    main()

