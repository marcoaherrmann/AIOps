"""
DelayPredict — FastAPI Inference Service
----------------------------------------
Milestone 01: Load model artifact, separate training from inference
Milestone 02: Define input/output contract with Pydantic schemas

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Test with:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"Airline":"WN","AirportFrom":"LAX","AirportTo":"SFO","DayOfWeek":3,"Length":60,"DepartureHour":8}'
"""

import sys
sys.path.append("/app/src")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from predict import load_model, build_input, predict as run_predict

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = "models/xgb_model.pkl"
model      = load_model()

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
def predict_endpoint(flight: FlightInput):
    """
    Predict whether a flight will be delayed.

    Returns the binary prediction and the model's delay probability.
    """
    try:
        input_df = build_input(
            airline       =flight.Airline,
            airport_from  =flight.AirportFrom,
            airport_to    =flight.AirportTo,
            day_of_week   =flight.DayOfWeek,
            length        =flight.Length,
            departure_hour=flight.DepartureHour,
        )
        result = run_predict(model, input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionOutput(
        delay_predicted   =result["delay_predicted"],
        delay_probability =result["delay_probability"],
        model             =MODEL_PATH,
    )