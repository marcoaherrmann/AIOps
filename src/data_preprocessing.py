"""
src/data_preprocessing.py
--------------------------
Loads the raw airline delay dataset and applies feature engineering.
Used by src/train.py and src/evaluate.py.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_PATH     = Path("data/raw/airlines_delay.csv")
RANDOM_STATE = 42

CATEGORICAL  = ["Airline", "AirportFrom", "AirportTo", "Route", "DayOfWeek"]
NUMERIC      = ["Length", "DepartureHour"]
FEATURES     = CATEGORICAL + NUMERIC
TARGET       = "Delay"


def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV and apply feature engineering."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Please place airlines_delay.csv in data/raw/"
        )

    df = pd.read_csv(path)

    # Feature engineering — same as in 03b_xgboost.ipynb
    df["DepartureHour"] = df["Time"] // 60
    df["Route"]         = df["AirportFrom"] + "-" + df["AirportTo"]
    df                  = df.drop(columns=["id", "Flight", "Time"])

    print(f"Loaded: {path} | shape: {df.shape}")
    return df


def get_splits(df: pd.DataFrame, test_size: float = 0.2):
    """Split into train/test sets, stratified by target."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train: {X_train.shape} | delay rate: {y_train.mean():.3f}")
    print(f"Test : {X_test.shape}  | delay rate: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = get_splits(df)
    print("\nFeatures:", FEATURES)
    print("Target  :", TARGET)