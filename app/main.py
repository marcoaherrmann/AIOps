"""
DelayPredict — FastAPI Inference Service
----------------------------------------
Milestone 01: Load model artifact, separate training from inference
Milestone 02: Define input/output contract with Pydantic schemas

Start with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Test with:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"Airline":"WN","AirportFrom":"LAX","AirportTo":"SFO","DayOfWeek":3,"Length":60,"DepartureHour":8}'
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ── Required for unpickling the XGBoost pipeline ─────────────────────────────
# The XGBoost pipeline uses a FunctionTransformer with this function.
# It must be defined here so joblib can find it when loading the model.
def cast_categoricals(X):
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    return X

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = "models/xgb_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        "Run notebook 03b_xgboost.ipynb first to generate the model artifact."
    )

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DelayPredict",
    description="Binary flight delay prediction — AIOps SoSe 2026",
    version="1.0.0",    
)

# ── Input schema (Milestone 02) ───────────────────────────────────────────────

class FlightInput(BaseModel):
    Airline: str = Field(
        ...,
        description="IATA airline code (e.g. 'WN', 'AA', 'DL')",
        examples=["WN"]
    )
    AirportFrom: str = Field(
        ...,
        description="IATA departure airport code (e.g. 'LAX')",
        examples=["LAX"]
    )
    AirportTo: str = Field(
        ...,
        description="IATA arrival airport code (e.g. 'SFO')",
        examples=["SFO"]
    )
    DayOfWeek: int = Field(
        ...,
        ge=1,
        le=7,
        description="Day of week — 1=Monday, 7=Sunday",
        examples=[3]
    )
    Length: int = Field(
        ...,
        gt=0,
        description="Scheduled flight duration in minutes",
        examples=[60]
    )
    DepartureHour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Scheduled departure hour (0–23)",
        examples=[8]
    )

    @field_validator("Airline", "AirportFrom", "AirportTo")
    @classmethod
    def must_be_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError("Must be uppercase IATA code (e.g. 'WN', 'LAX')")
        return v

# ── Output schema (Milestone 02) ──────────────────────────────────────────────
class PredictionOutput(BaseModel):
    delay_predicted: bool = Field(
        ...,
        description="True if the flight is predicted to be delayed"
    )
    delay_probability: float = Field(
        ...,
        description="Model confidence that the flight will be delayed (0.0–1.0)"
    )
    model: str = Field(
        ...,
        description="Model used for inference"
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Check that the service and model are loaded correctly."""
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionOutput)
def predict(flight: FlightInput):
    """
    Predict whether a flight will be delayed.

    Returns the binary prediction and the model's delay probability.
    """
    # Build a single-row DataFrame matching training feature order
    input_df = pd.DataFrame([{
        "Airline"      : flight.Airline,
        "AirportFrom"  : flight.AirportFrom,
        "AirportTo"    : flight.AirportTo,
        "Route"        : f"{flight.AirportFrom}-{flight.AirportTo}",
        "DayOfWeek"    : flight.DayOfWeek,
        "Length"       : flight.Length,
        "DepartureHour": flight.DepartureHour,
    }])

    try:
        delay_prob = float(model.predict_proba(input_df)[0, 1])
        delay_pred = bool(model.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionOutput(
        delay_predicted=delay_pred,
        delay_probability=round(delay_prob, 4),
        model=MODEL_PATH,
    )