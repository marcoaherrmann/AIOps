"""
src/seed_predictions.py
------------------------
Seeds the predictions table with a sample from the raw dataset.
Run once to populate Metabase charts with realistic data.

Usage:
    docker exec aiops-api-1 python src/seed_predictions.py
    docker exec aiops-api-1 python src/seed_predictions.py --rows 2000
"""

import sys
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

sys.path.append("/app/src")

from data_preprocessing import FEATURES, TARGET
from database import log_prediction

RAW_DATA_PATH = Path("/app/data/raw/airlines_delay.csv")
MODEL_PATH    = Path("/app/models/xgb_model.pkl")
DEFAULT_ROWS  = 1000


def seed(n_rows: int = DEFAULT_ROWS):
    print(f"Loading dataset and model...")
    df    = pd.read_csv(RAW_DATA_PATH)
    model = joblib.load(MODEL_PATH)

    df["DepartureHour"] = df["Time"] // 60
    df["Route"]         = df["AirportFrom"] + "-" + df["AirportTo"]

    # Stratified sample by DepartureHour — ensures every hour has enough rows
    # so Metabase delay-rate chart doesn't spike on hours with 1-2 data points
    min_per_hour = max(30, n_rows // df["DepartureHour"].nunique())
    parts = []
    for hour, group in df.groupby("DepartureHour"):
        parts.append(group.sample(n=min(len(group), min_per_hour), random_state=42))
    df_stratified = pd.concat(parts).sample(frac=1, random_state=42)
    sample = df_stratified.sample(n=min(n_rows, len(df_stratified)), random_state=42).reset_index(drop=True)

    X = sample[FEATURES]
    probs      = model.predict_proba(X)[:, 1]
    predicted  = (probs >= 0.5).astype(bool)

    # Spread timestamps over the last 7 days so charts look alive over time
    base_time = datetime.utcnow()
    print(f"Writing {len(sample)} predictions to DB...")

    for i, row in sample.iterrows():
        ts = base_time - timedelta(
            days=random.uniform(0, 7),
            hours=random.uniform(0, 24),
        )
        try:
            log_prediction(
                airline          =str(row["Airline"]),
                airport_from     =str(row["AirportFrom"]),
                airport_to       =str(row["AirportTo"]),
                day_of_week      =int(row["DayOfWeek"]),
                departure_hour   =int(row["DepartureHour"]),
                length           =int(row["Length"]),
                delay_predicted  =bool(predicted[i]),
                delay_probability=float(probs[i]),
            )
        except Exception as e:
            print(f"Row {i} failed: {e}")

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(sample)} done...")

    print(f"Done — {len(sample)} predictions seeded.")


if __name__ == "__main__":
    n = DEFAULT_ROWS
    for arg in sys.argv[1:]:
        if arg.startswith("--rows="):
            n = int(arg.split("=")[1])
        elif arg == "--rows" and len(sys.argv) > sys.argv.index(arg) + 1:
            n = int(sys.argv[sys.argv.index(arg) + 1])
    seed(n)
