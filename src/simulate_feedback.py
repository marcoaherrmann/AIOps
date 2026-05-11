"""
src/simulate_feedback.py
-------------------------
Simulates real-world feedback by matching predictions in predictions.csv
against the actual labels from the test set.

In a real system this would happen automatically when flight outcomes
become known. Here we use the held-out test set as a stand-in.

Usage:
    python src/simulate_feedback.py
    python src/simulate_feedback.py --n 50   # simulate 50 feedbacks
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.append("/app/src")
from data_preprocessing import load_data, get_splits, FEATURES

PRED_LOG_PATH = Path("data/processed/predictions.csv")


def simulate(n: int = 20):
    """Fill actual_delay for the last n predictions without feedback."""

    if not PRED_LOG_PATH.exists():
        print("No predictions logged yet. Run /predict first.")
        return

    # ── Load predictions ───────────────────────────────────────────────────────
    preds = pd.read_csv(PRED_LOG_PATH, dtype=str)
    missing = preds[preds["actual_delay"].isna()]

    if missing.empty:
        print("All predictions already have feedback.")
        return

    to_fill = missing.head(n)
    print(f"Filling feedback for {len(to_fill)} predictions...")

    # ── Load test set — same split as training ─────────────────────────────────
    df = load_data()
    _, X_test, _, y_test = get_splits(df)
    test_df = X_test.copy()
    test_df["Delay"] = y_test.values

    # ── Match each prediction to a test row ────────────────────────────────────
    filled = 0
    for idx, row in to_fill.iterrows():
        # Find a matching row in the test set by flight features
        match = test_df[
            (test_df["Airline"]       == row["Airline"]) &
            (test_df["AirportFrom"]   == row["AirportFrom"]) &
            (test_df["AirportTo"]     == row["AirportTo"]) &
            (test_df["DayOfWeek"]     == row["DayOfWeek"]) &
            (test_df["Length"].astype(str) == row["Length"]) &
            (test_df["DepartureHour"].astype(str) == row["DepartureHour"])
        ]

        if not match.empty:
            actual = int(match.iloc[0]["Delay"])
        else:
            # No exact match — sample randomly from test set
            actual = int(test_df["Delay"].sample(1, random_state=None).values[0])

        preds.loc[idx, "actual_delay"] = str(actual)
        filled += 1

    preds.to_csv(PRED_LOG_PATH, index=False)
    print(f"Done — filled {filled} feedbacks.")
    print(f"Predictions log: {PRED_LOG_PATH}")

    # ── Show summary ───────────────────────────────────────────────────────────
    preds_updated = pd.read_csv(PRED_LOG_PATH, dtype=str)
    has_feedback  = preds_updated[preds_updated["actual_delay"].notna()]
    print(f"\nTotal predictions : {len(preds_updated)}")
    print(f"With feedback     : {len(has_feedback)}")
    print(f"Without feedback  : {len(preds_updated) - len(has_feedback)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of feedbacks to simulate")
    args = parser.parse_args()
    simulate(n=args.n)