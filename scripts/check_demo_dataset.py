from pathlib import Path
import pandas as pd

DATA_FILE = Path("data/demo/smartcart_demo_events.csv")

REQUIRED_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

def main():
    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        print("ERROR: Missing columns:", missing)
        return

    print("Demo CSV schema: OK")
    print("Total events:", len(df))
    print("Unique sessions:", df["user_session"].nunique())
    print("\nEvent counts:")
    print(df["event_type"].value_counts().to_string())
    print("\nNOTE: This is synthetic demo data.")
    print("Use it for setup checks, not official model evaluation.")

if __name__ == "__main__":
    main()
