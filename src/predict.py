"""
src/predict.py
---------------
Single-flight prediction using the trained XGBoost model.
Used by app/main.py for the FastAPI inference service.

Usage:
    python src/predict.py
"""

import joblib
import pandas as pd
from pathlib import Path

from data_preprocessing import FEATURES, CATEGORICAL, NUMERIC

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/xgb_model.pkl")


def load_model(path: Path = MODEL_PATH):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            "Run: python src/train.py first."
        )
    return joblib.load(path)


def build_input(
    airline: str,
    airport_from: str,
    airport_to: str,
    day_of_week: int,
    length: int,
    departure_hour: int,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature order."""
    return pd.DataFrame([{
        "Airline"      : airline,
        "AirportFrom"  : airport_from,
        "AirportTo"    : airport_to,
        "Route"        : f"{airport_from}-{airport_to}",
        "DayOfWeek"    : day_of_week,
        "Length"       : length,
        "DepartureHour": departure_hour,
    }])


def predict(model, input_df: pd.DataFrame) -> dict:
    """Run prediction and return delay flag + probability."""
    delay_prob = float(model.predict_proba(input_df)[0, 1])
    delay_pred = bool(model.predict(input_df)[0])

    return {
        "delay_predicted"  : delay_pred,
        "delay_probability": round(delay_prob, 4),
    }


if __name__ == "__main__":
    # Example prediction
    model = load_model()

    input_df = build_input(
        airline       = "WN",
        airport_from  = "LAX",
        airport_to    = "SFO",
        day_of_week   = 3,
        length        = 60,
        departure_hour= 8,
    )

    result = predict(model, input_df)
    print("Input:")
    print(input_df.to_string(index=False))
    print("\nPrediction:")
    print(f"  Delayed       : {result['delay_predicted']}")
    print(f"  Probability   : {result['delay_probability']:.4f}")